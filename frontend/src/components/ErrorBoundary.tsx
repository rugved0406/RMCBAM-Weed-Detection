import { Component, type ErrorInfo, type ReactNode } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('WeedVision UI uncaught error:', error, errorInfo);
  }

  public handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div
          style={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '40px',
            background: 'var(--fo-bg-dark)',
            color: 'var(--fo-text-main)',
            textAlign: 'center',
          }}
        >
          <div
            style={{
              maxWidth: '520px',
              background: 'var(--fo-bg-panel)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              borderRadius: '8px',
              padding: '24px',
              boxShadow: '0 8px 30px rgba(0,0,0,0.5)',
            }}
          >
            <AlertTriangle size={36} style={{ color: '#ef4444', marginBottom: '12px' }} />
            <h3 style={{ fontSize: '16px', fontWeight: 700, marginBottom: '8px' }}>
              Something went wrong in this view
            </h3>
            <p style={{ fontSize: '12px', color: 'var(--fo-text-subtle)', marginBottom: '16px' }}>
              {this.state.error?.message || 'An unexpected rendering error occurred.'}
            </p>
            <div style={{ display: 'flex', gap: '10px', justifyContent: 'center' }}>
              <button
                className="fo-btn fo-btn-primary"
                onClick={this.handleReset}
                style={{ fontSize: '12px', padding: '6px 14px' }}
              >
                <RefreshCw size={13} /> Try Again
              </button>
              <button
                className="fo-btn fo-btn-secondary"
                onClick={() => window.location.reload()}
                style={{ fontSize: '12px', padding: '6px 14px' }}
              >
                Reload Application
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
