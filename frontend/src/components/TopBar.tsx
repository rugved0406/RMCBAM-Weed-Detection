import React from 'react';
import type { SystemHealth } from '../types';
import { Cpu, Zap, ShieldCheck, CheckCircle2, AlertTriangle, Sprout } from 'lucide-react';

interface Props {
  health: SystemHealth | null;
  activeCheckpoint: string | null;
  isRmcbam?: boolean;
}

export const TopBar: React.FC<Props> = ({ health, activeCheckpoint, isRmcbam }) => {
  const modelFilename = activeCheckpoint ? activeCheckpoint.split(/[/\\]/).pop() : 'weed_detection_best.pt';

  return (
    <header className="fo-topbar">
      {/* Brand & Subtitle */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '28px',
            height: '28px',
            borderRadius: '6px',
            background: 'var(--fo-accent-green-light)',
            color: 'var(--fo-accent-green)',
            border: '1px solid var(--fo-accent-green-border)',
          }}
        >
          <Sprout size={16} />
        </div>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontWeight: 700, fontSize: '14px', letterSpacing: '-0.01em', color: 'var(--fo-text-main)' }}>
              WeedVision
            </span>
            <span
              style={{
                fontSize: '11px',
                color: isRmcbam ? 'var(--fo-accent-green)' : 'var(--fo-text-muted)',
                background: isRmcbam ? 'var(--fo-accent-green-light)' : 'var(--fo-bg-elevated)',
                padding: '1px 6px',
                borderRadius: '4px',
                border: isRmcbam ? '1px solid var(--fo-accent-green-border)' : '1px solid var(--fo-border)',
                fontFamily: 'var(--font-mono)',
                fontWeight: 600,
              }}
            >
              {isRmcbam ? 'YOLOv8s + RMCBAM' : 'YOLOv8s Baseline'}
            </span>
          </div>
          <div style={{ fontSize: '10px', color: 'var(--fo-text-muted)', marginTop: '-1px' }}>
            Agricultural Computer Vision & Field Inspection
          </div>
        </div>
      </div>

      {/* Hardware & Diagnostics Status */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '14px', fontSize: '12px' }}>
        {health?.demo_mode && (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              color: '#b45309',
              background: '#fef3c7',
              padding: '2px 8px',
              borderRadius: '4px',
              border: '1px solid #fde68a',
              fontSize: '11px',
              fontWeight: 600,
            }}
          >
            <AlertTriangle size={12} />
            <span>DEMO MODE</span>
          </div>
        )}

        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            color: 'var(--fo-text-muted)',
            background: 'var(--fo-bg-elevated)',
            border: '1px solid var(--fo-border)',
            padding: '3px 8px',
            borderRadius: '4px',
          }}
        >
          <Cpu size={13} style={{ color: 'var(--fo-text-subtle)' }} />
          <span>Device:</span>
          <span style={{ color: 'var(--fo-text-main)', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
            {health?.device || 'cpu'}
          </span>
          {health?.cuda_available && (
            <span style={{ color: 'var(--fo-accent-green)', display: 'inline-flex', alignItems: 'center', marginLeft: '4px' }}>
              <Zap size={11} /> CUDA
            </span>
          )}
        </div>

        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            color: 'var(--fo-text-muted)',
            background: 'var(--fo-bg-elevated)',
            border: '1px solid var(--fo-border)',
            padding: '3px 8px',
            borderRadius: '4px',
          }}
        >
          <ShieldCheck size={13} style={{ color: 'var(--fo-accent-green)' }} />
          <span>Active Checkpoint:</span>
          <span style={{ color: 'var(--fo-text-main)', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
            {modelFilename}
          </span>
        </div>

        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '5px',
            color: 'var(--fo-accent-green)',
            fontSize: '11px',
            fontWeight: 600,
            padding: '3px 6px',
          }}
        >
          <CheckCircle2 size={13} />
          <span>ONLINE</span>
        </div>
      </div>
    </header>
  );
};
