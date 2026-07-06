from backend.router.advisor_result import AdvisorResult

class AdvisorInterface:
    """Interface that must be implemented by all specialized advisor domains."""
    async def evaluate_fit(self, query: str, context: dict) -> float:
        """Returns fit score between 0.0 (completely irrelevant) and 1.0 (perfect match)."""
        raise NotImplementedError

    async def execute(self, query: str, context: dict) -> AdvisorResult:
        """Executes advisor-specific logic and returns standard AdvisorResult."""
        raise NotImplementedError

    def metadata(self) -> dict:
        """Returns metadata configuration details for this advisor."""
        raise NotImplementedError
