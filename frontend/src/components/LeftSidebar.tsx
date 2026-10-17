import {
  Crosshair,
  BarChart3,
  MapPin,
  Flame,
  GitCompare,
  History,
  Sliders,
  Database,
} from 'lucide-react';

export type WorkspaceId =
  | 'detection'
  | 'analysis'
  | 'localization'
  | 'explainability'
  | 'dataset'
  | 'comparison'
  | 'history'
  | 'settings';

interface Props {
  activeWorkspace: WorkspaceId;
  onSelectWorkspace: (id: WorkspaceId) => void;
  historyCount?: number;
}

export const LeftSidebar: React.FC<Props> = ({
  activeWorkspace,
  onSelectWorkspace,
  historyCount = 0,
}) => {
  const navItems: { id: WorkspaceId; label: string; icon: React.ReactNode; badge?: number | string }[] = [
    {
      id: 'detection',
      label: 'Weed Detection',
      icon: <Crosshair size={16} />,
    },
    {
      id: 'analysis',
      label: 'Weed Analysis',
      icon: <BarChart3 size={16} />,
    },
    {
      id: 'localization',
      label: 'Weed Localization',
      icon: <MapPin size={16} />,
    },
    {
      id: 'explainability',
      label: 'Weed Explainability',
      icon: <Flame size={16} />,
    },
    {
      id: 'dataset',
      label: 'Dataset Analytics',
      icon: <Database size={16} />,
    },
    {
      id: 'comparison',
      label: 'Model Comparison',
      icon: <GitCompare size={16} />,
    },
    {
      id: 'history',
      label: 'Analysis History',
      icon: <History size={16} />,
      badge: historyCount > 0 ? historyCount : undefined,
    },
    {
      id: 'settings',
      label: 'Settings',
      icon: <Sliders size={16} />,
    },
  ];

  return (
    <aside className="fo-sidebar">
      <div
        style={{
          padding: '6px 12px 10px',
          fontSize: '11px',
          fontWeight: 700,
          color: 'var(--fo-text-muted)',
          letterSpacing: '0.06em',
          textTransform: 'uppercase',
        }}
      >
        Workspaces
      </div>

      <nav style={{ display: 'flex', flexDirection: 'column', gap: '3px', flex: 1 }}>
        {navItems.map((item) => {
          const isActive = activeWorkspace === item.id;
          return (
            <button
              key={item.id}
              className={`fo-nav-button ${isActive ? 'active' : ''}`}
              onClick={() => onSelectWorkspace(item.id)}
              title={item.label}
            >
              <span style={{ display: 'flex', alignItems: 'center', color: isActive ? 'var(--fo-accent-green)' : 'var(--fo-text-subtle)' }}>
                {item.icon}
              </span>
              <span className="fo-nav-label" style={{ flex: 1 }}>
                {item.label}
              </span>
              {item.badge !== undefined && (
                <span
                  style={{
                    fontSize: '10px',
                    fontFamily: 'var(--font-mono)',
                    fontWeight: 700,
                    padding: '1px 6px',
                    borderRadius: '999px',
                    background: isActive ? 'var(--fo-accent-green)' : 'var(--fo-border)',
                    color: isActive ? '#ffffff' : 'var(--fo-text-subtle)',
                  }}
                >
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Understated scientific footer */}
      <div
        style={{
          padding: '12px',
          borderTop: '1px solid var(--fo-border)',
          marginTop: 'auto',
          fontSize: '11px',
          color: 'var(--fo-text-muted)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600, color: 'var(--fo-text-subtle)', marginBottom: '2px' }}>
          <Database size={12} style={{ color: 'var(--fo-accent-green)' }} /> RMCBAM Attention
        </div>
        <div>Ultralytics YOLOv8s Pipeline</div>
      </div>
    </aside>
  );
};
