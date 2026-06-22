"use client";
import React, { Component, ReactNode, ErrorInfo } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

/**
 * Error Boundary component that catches JavaScript errors anywhere in the
 * child component tree and displays a fallback UI instead of crashing.
 */
export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    // Update state so the next render will show the fallback UI
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    // Log error to console (could also send to error tracking service)
    console.error("[Farm360 ErrorBoundary] Caught error:", error);
    console.error("[Farm360 ErrorBoundary] Component stack:", errorInfo.componentStack);
  }

  handleReset = (): void => {
    this.setState({ hasError: false, error: null });
  };

  render(): ReactNode {
    if (this.state.hasError) {
      // Custom fallback UI
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // Default error UI
      return (
        <div
          className="flex flex-col items-center justify-center min-h-[200px] p-6 rounded-2xl mx-auto max-w-md"
          style={{
            background: "var(--surface)",
            border: "1px solid rgba(239, 68, 68, 0.3)",
          }}
        >
          <div
            className="w-12 h-12 rounded-full flex items-center justify-center mb-4"
            style={{ background: "rgba(239, 68, 68, 0.15)" }}
          >
            <AlertTriangle size={24} style={{ color: "#ef4444" }} />
          </div>

          <h2
            className="text-lg font-semibold mb-2"
            style={{ color: "#f2f2f2" }}
          >
            Something went wrong
          </h2>

          <p
            className="text-sm text-center mb-4 max-w-sm"
            style={{ color: "#888" }}
          >
            The chat encountered an unexpected error. Your conversation is safe.
          </p>

          {this.state.error && (
            <details className="mb-4 w-full max-w-sm">
              <summary
                className="text-xs cursor-pointer hover:text-white transition-colors"
                style={{ color: "#666" }}
              >
                Error details
              </summary>
              <pre
                className="mt-2 p-3 rounded-lg text-xs overflow-x-auto"
                style={{
                  background: "#111",
                  color: "#f87171",
                  border: "1px solid var(--border)",
                }}
              >
                {this.state.error.message}
              </pre>
            </details>
          )}

          <button
            onClick={this.handleReset}
            className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all hover:scale-105"
            style={{
              background: "var(--accent)",
              color: "white",
            }}
          >
            <RefreshCw size={14} />
            Try Again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
