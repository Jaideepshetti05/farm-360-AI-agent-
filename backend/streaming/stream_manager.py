import time
import re
import uuid
import asyncio
from typing import AsyncGenerator, Dict, Any, Optional
from loguru import logger

from backend.services.cache_service import CacheService
from backend.streaming.interfaces import IStreamManager
from backend.streaming.stream_context import StreamContext, StreamState
from backend.streaming.stream_events import StreamEvent
from backend.streaming.stream_metrics import StreamMetrics
from backend.streaming.stream_response import SSETransportAdapter, TransportAdapter
from backend.validator.engine import ValidationEngine
from backend.router.router import IntentRouter
from backend.provider_manager import provider_manager
from backend.services.context_builder import PromptContextService
from backend.observability import TraceContext, EventBus, log_structured

class StreamManager(IStreamManager):
    def __init__(self, transport_adapter: Optional[TransportAdapter] = None):
        self.validator_engine = ValidationEngine()
        self.transport_adapter = transport_adapter or SSETransportAdapter()

    def _validate_chunk(self, chunk: str) -> bool:
        """Lightweight per-token regex checks for secrets/leaks."""
        key_patterns = [
            r"sk-[a-zA-Z0-9]{10,}",
            r"AIzaSy[a-zA-Z0-9_-]{10}"
        ]
        for pattern in key_patterns:
            if re.search(pattern, chunk):
                return False
        return True

    async def stream_query(
        self,
        query: str,
        context: Dict[str, Any],
        prompt_key: str = "general_assistant",
        request_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Coordinates connection streaming, cache checks, per-token sanitizers,
        and post-stream validator execution. Yields serialized transport-adapted events.
        """
        user_profile = context.get("user_profile") or {}
        language = context.get("language") or "en"
        prompt_version = context.get("prompt_version") or "1.0.0"
        model = context.get("model") or "gemini-2.5-flash"
        provider = context.get("provider") or "gemini"

        # Create/resolve TraceContext
        trace_ctx = context.get("trace_context")
        if not trace_ctx:
            trace_ctx = TraceContext(
                request_id=request_id or context.get("request_id") or str(uuid.uuid4()),
                session_id=context.get("session_id", "default_session"),
                provider=provider,
                model=model,
                prompt_version=prompt_version
            )
        request_id = trace_ctx.request_id

        # Publish Workflow Started Event
        EventBus.publish("workflow_started", {
            "request_id": request_id,
            "query": query,
            "trace_context": trace_ctx.model_dump()
        })

        log_structured(
            level="info",
            component="StreamingManager",
            operation="stream_query_start",
            message="Starting query stream.",
            trace_ctx=trace_ctx
        )

        # Initialize Stream Context and Metrics
        stream_ctx = StreamContext(
            request_id=request_id,
            session_id=trace_ctx.session_id,
            query=query,
            user_profile=user_profile,
            language=language,
            prompt_version=prompt_version,
            model=model,
            provider=provider
        )
        
        metrics = StreamMetrics(
            start_time=stream_ctx.start_time,
            provider=provider,
            model=model,
            state_transitions=stream_ctx.state_transitions
        )

        sequence = 0

        def emit(event_type: str, payload: Any, is_terminal: bool = False, metadata: Optional[Dict[str, Any]] = None) -> str:
            nonlocal sequence
            meta = metadata or {}
            event = StreamEvent(
                event_type=event_type,
                stream_id=request_id,
                sequence=sequence,
                provider=provider,
                model=model,
                payload=payload,
                metadata={**meta, "current_state": stream_ctx.current_state},
                is_terminal=is_terminal
            )
            sequence += 1
            return self.transport_adapter.serialize(event)

        # ── State: Initializing ──────────────────────────────────────────────
        stream_ctx.transition_to("Initializing")
        log_structured(
            level="info",
            component="StreamingManager",
            operation="state_transition",
            message="State transition: Initializing",
            trace_ctx=trace_ctx
        )
        yield emit("start", {"message": "Stream initialized"})

        # ── State: CacheLookup ───────────────────────────────────────────────
        stream_ctx.transition_to("CacheLookup")
        log_structured(
            level="info",
            component="StreamingManager",
            operation="state_transition",
            message="State transition: CacheLookup",
            trace_ctx=trace_ctx
        )
        
        cache_key = CacheService.generate_key(
            prompt_version=prompt_version,
            model=model,
            provider=provider,
            language=language,
            context={"query": query, "profile": user_profile}
        )
        
        EventBus.publish("cache_lookup_started", {
            "request_id": request_id,
            "cache_key": cache_key,
            "trace_context": trace_ctx.model_dump()
        })

        try:
            cached = CacheService.get(cache_key, request_id=request_id)
        except Exception as ce:
            log_structured(
                level="error",
                component="StreamingManager",
                operation="cache_get",
                message=f"Cache read error: {ce}",
                trace_ctx=trace_ctx,
                status="ERROR",
                metadata={"error": str(ce)}
            )
            cached = None

        if cached:
            log_structured(
                level="info",
                component="StreamingManager",
                operation="cache_hit",
                message="Cache hit. Yielding cached response.",
                trace_ctx=trace_ctx
            )
            metrics.record_first_token()
            
            # Simulated fast token stream for cached entries
            words = cached.split(" ")
            for idx, word in enumerate(words):
                chunk = word + (" " if idx < len(words) - 1 else "")
                metrics.increment_tokens(len(chunk.split()))
                yield emit("token", chunk)
                await asyncio.sleep(0.005)

            stream_ctx.transition_to("Completed")
            metrics.finish()
            
            log_structured(
                level="info",
                component="StreamingManager",
                operation="state_transition",
                message="State transition: Completed (via cache)",
                trace_ctx=trace_ctx,
                duration_ms=metrics.total_duration_ms,
                status="SUCCESS",
                metadata={
                    "ttft_ms": metrics.ttft_ms,
                    "tokens_per_second": metrics.tokens_per_second,
                    "token_count": metrics.token_count
                }
            )
            
            EventBus.publish("workflow_completed", {
                "request_id": request_id,
                "metrics": metrics.model_dump(),
                "trace_context": trace_ctx.model_dump(),
                "cached": True
            })
            
            yield emit("complete", {
                "metrics": {
                    "ttft_ms": metrics.ttft_ms,
                    "total_duration_ms": metrics.total_duration_ms,
                    "tokens_per_second": metrics.tokens_per_second,
                    "token_count": metrics.token_count,
                    "cached": True
                }
            }, is_terminal=True)
            return

        # ── State: Routing ───────────────────────────────────────────────────
        stream_ctx.transition_to("Routing")
        log_structured(
            level="info",
            component="StreamingManager",
            operation="state_transition",
            message="State transition: Routing",
            trace_ctx=trace_ctx
        )
        
        EventBus.publish("router_started", {
            "request_id": request_id,
            "query": query,
            "trace_context": trace_ctx.model_dump()
        })

        router = IntentRouter()
        advisors = router.registry.get_advisors()
        
        try:
            # Concurrently evaluate fit scores
            tasks = [adv.evaluate_fit(query, context) for adv in advisors]
            eval_results = await asyncio.gather(*tasks)
            
            best_adv = None
            best_score = -1.0
            for adv, score in zip(advisors, eval_results):
                if score > best_score:
                    best_score = score
                    best_adv = adv
            
            if best_score < 0.30:
                best_adv = router.registry.get_by_name("GeneralAdvisor")
                best_score = 0.25

            log_structured(
                level="info",
                component="IntentRouter",
                operation="route_select",
                message=f"Routed to {best_adv.name} (score: {best_score:.2f})",
                trace_ctx=trace_ctx,
                status="SUCCESS",
                metadata={"advisor": best_adv.name, "score": best_score}
            )
            
            EventBus.publish("router_completed", {
                "request_id": request_id,
                "advisor": best_adv.name,
                "score": best_score,
                "trace_context": trace_ctx.model_dump()
            })
            
            yield emit("progress", {"status": f"Routing completed to {best_adv.name}"})

            # Retrieve RAG context if applicable
            rag_context = ""
            if best_adv.name in ["CropAdvisor", "DiseaseAdvisor", "AnimalAdvisor", "WeatherAdvisor", "MarketAdvisor"]:
                try:
                    from backend.rag import RAGService
                    rag_service = RAGService()
                    
                    EventBus.publish("rag_started", {"request_id": request_id, "query": query})
                    rag_start = time.time()
                    rag_context = await rag_service.get_context(query)
                    rag_duration = (time.time() - rag_start) * 1000.0
                    
                    EventBus.publish("rag_completed", {"request_id": request_id, "duration_ms": rag_duration})
                except Exception as re:
                    log_structured(
                        level="warning",
                        component="RAGService",
                        operation="get_context",
                        message=f"RAG lookup skipped: {re}",
                        trace_ctx=trace_ctx,
                        status="WARNING"
                    )
            
            ml_context = context.get("ml_context", "")
            if rag_context:
                ml_context = f"{ml_context}\n\n{rag_context}".strip()

            # Build messages list
            messages = PromptContextService.build_prompt_context(
                system_prompt_template=best_adv.prompt_key,
                user_profile=user_profile,
                ml_context=ml_context,
                recent_history=context.get("recent_history", []),
                summary_text=context.get("summary_text"),
                max_context_tokens=4096,
                query=query
            )
            messages.append({"role": "user", "content": query})

        except Exception as e:
            log_structured(
                level="error",
                component="StreamingManager",
                operation="setup",
                message=f"Routing/Setup phase failed: {e}",
                trace_ctx=trace_ctx,
                status="ERROR",
                metadata={"error": str(e)}
            )
            stream_ctx.transition_to("Failed")
            yield emit("error", f"Routing error: {str(e)}", is_terminal=True)
            return

        # ── State: Generating ────────────────────────────────────────────────
        stream_ctx.transition_to("Generating")
        log_structured(
            level="info",
            component="StreamingManager",
            operation="state_transition",
            message="State transition: Generating",
            trace_ctx=trace_ctx
        )

        assembled_chunks = []
        gen = None
        try:
            gen = provider_manager.stream_completion_async(
                messages=messages,
                request_id=request_id,
                temperature=context.get("temperature", 0.7),
                max_tokens=context.get("max_tokens", 2048)
            )

            async for token in gen:
                metrics.record_first_token()
                
                # Check for secrets/leaks
                if not self._validate_chunk(token):
                    log_structured(
                        level="error",
                        component="StreamingManager",
                        operation="validation_per_token",
                        message="Suspicious signature detected in token chunk. Aborting stream.",
                        trace_ctx=trace_ctx,
                        status="BLOCK"
                    )
                    stream_ctx.transition_to("Failed")
                    yield emit("error", "Safety validation exception: Suspicious token pattern.", is_terminal=True)
                    return

                assembled_chunks.append(token)
                metrics.increment_tokens(len(token.split()))
                yield emit("token", token)

        except asyncio.CancelledError:
            log_structured(
                level="warning",
                component="StreamingManager",
                operation="generation",
                message="Generation cancelled by client.",
                trace_ctx=trace_ctx,
                status="CANCELLED"
            )
            stream_ctx.transition_to("Cancelled")
            yield emit("error", "Request cancelled by client.", is_terminal=True)
            raise
        except Exception as ge:
            log_structured(
                level="error",
                component="StreamingManager",
                operation="generation",
                message=f"Generation exception: {ge}",
                trace_ctx=trace_ctx,
                status="ERROR",
                metadata={"error": str(ge)}
            )
            stream_ctx.transition_to("Failed")
            yield emit("error", f"LLM generation failed: {str(ge)}", is_terminal=True)
            return
        finally:
            if gen is not None:
                await gen.aclose()

        assembled_text = "".join(assembled_chunks)

        # ── State: Validating ────────────────────────────────────────────────
        stream_ctx.transition_to("Validating")
        log_structured(
            level="info",
            component="StreamingManager",
            operation="state_transition",
            message="State transition: Validating",
            trace_ctx=trace_ctx
        )

        EventBus.publish("validator_started", {
            "request_id": request_id,
            "text_len": len(assembled_text),
            "trace_context": trace_ctx.model_dump()
        })

        try:
            validated_text, results = await self.validator_engine.validate_response(assembled_text, context)
            is_blocked = any(r.status == "BLOCK" for r in results)
            
            log_structured(
                level="info",
                component="ValidationEngine",
                operation="validation_check",
                message=f"Validation completed. Block status: {is_blocked}",
                trace_ctx=trace_ctx,
                metadata={"results": [{"status": getattr(r, "status", None), "rule_name": getattr(r, "rule_name", None)} for r in results]}
            )
            
            EventBus.publish("validator_completed", {
                "request_id": request_id,
                "is_blocked": is_blocked,
                "trace_context": trace_ctx.model_dump()
            })

            if is_blocked:
                log_structured(
                    level="error",
                    component="ValidationEngine",
                    operation="validate_response",
                    message="Response blocked by Validator safety layer.",
                    trace_ctx=trace_ctx,
                    status="BLOCK"
                )
                stream_ctx.transition_to("Failed")
                yield emit("error", "Response blocked due to safety guidelines.", is_terminal=True)
                return
        except Exception as ve:
            log_structured(
                level="error",
                component="StreamingManager",
                operation="validate_phase",
                message=f"Validation phase failed: {ve}",
                trace_ctx=trace_ctx,
                status="ERROR",
                metadata={"error": str(ve)}
            )
            stream_ctx.transition_to("Failed")
            yield emit("error", f"Safety validation error: {str(ve)}", is_terminal=True)
            return

        # ── State: Saving ────────────────────────────────────────────────────
        stream_ctx.transition_to("Saving")
        log_structured(
            level="info",
            component="StreamingManager",
            operation="state_transition",
            message="State transition: Saving",
            trace_ctx=trace_ctx
        )

        # Save to Cache
        try:
            CacheService.set(cache_key, validated_text, request_id=request_id)
        except Exception as cse:
            log_structured(
                level="warning",
                component="StreamingManager",
                operation="cache_set",
                message=f"Failed to cache response: {cse}",
                trace_ctx=trace_ctx,
                status="WARNING"
            )

        # Save to DB history
        try:
            from backend.services.database_service import UnitOfWork
            from backend.models.database import ConversationHistory
            
            async def _save():
                async with UnitOfWork() as uow:
                    session_id = context.get("session_id", "default_session")
                    uow.session.add(ConversationHistory(session_id=session_id, role="user", content=query))
                    uow.session.add(ConversationHistory(session_id=session_id, role="assistant", content=validated_text))
                    await uow.session.commit()
            await _save()
        except Exception as dbe:
            log_structured(
                level="warning",
                component="StreamingManager",
                operation="db_history_persist",
                message=f"Failed to persist conversation history: {dbe}",
                trace_ctx=trace_ctx,
                status="WARNING"
            )

        # ── State: Completed ─────────────────────────────────────────────────
        stream_ctx.transition_to("Completed")
        metrics.finish()
        
        ttft_str = f"{metrics.ttft_ms:.1f}ms" if metrics.ttft_ms is not None else "N/A"
        duration_str = f"{metrics.total_duration_ms:.1f}ms" if metrics.total_duration_ms is not None else "N/A"
        
        log_structured(
            level="info",
            component="StreamingManager",
            operation="state_transition",
            message="State transition: Completed",
            trace_ctx=trace_ctx,
            duration_ms=metrics.total_duration_ms,
            status="SUCCESS",
            metadata={
                "ttft_ms": metrics.ttft_ms,
                "tokens_per_second": metrics.tokens_per_second,
                "token_count": metrics.token_count
            }
        )

        # Get timings metadata
        timings = metrics.get_state_timings()

        EventBus.publish("workflow_completed", {
            "request_id": request_id,
            "metrics": metrics.model_dump(),
            "trace_context": trace_ctx.model_dump(),
            "cached": False
        })

        yield emit("complete", {
            "metrics": {
                "ttft_ms": metrics.ttft_ms,
                "total_duration_ms": metrics.total_duration_ms,
                "tokens_per_second": metrics.tokens_per_second,
                "token_count": metrics.token_count,
                "cached": False,
                "timings": timings
            }
        }, is_terminal=True)
