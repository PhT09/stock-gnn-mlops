import type { HTMLAttributes, ReactNode } from 'react'
import { cn } from '@/lib/utils'

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: 'default' | 'success' | 'danger' | 'warning' | 'outline' | 'muted'
  children: ReactNode
}

export function Badge({ variant = 'default', className, children, ...props }: BadgeProps) {
  const styles: Record<string, string> = {
    default: 'bg-primary text-primary-foreground',
    success: 'bg-success/15 text-success border border-success/30',
    danger: 'bg-danger/15 text-danger border border-danger/30',
    warning: 'bg-warning/15 text-warning border border-warning/30',
    muted: 'bg-muted text-muted-foreground',
    outline: 'border border-border text-foreground',
  }
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium',
        styles[variant],
        className
      )}
      {...props}
    >
      {children}
    </span>
  )
}

export function ConfidenceBadge({ level }: { level: 'HIGH' | 'MEDIUM' | 'LOW' }) {
  const variant =
    level === 'HIGH' ? 'success' : level === 'MEDIUM' ? 'warning' : 'muted'
  return <Badge variant={variant}>{level}</Badge>
}

export function SignalBadge({ prediction }: { prediction: 0 | 1 }) {
  return prediction === 1 ? (
    <Badge variant="success">↑ TĂNG</Badge>
  ) : (
    <Badge variant="danger">↓ GIẢM</Badge>
  )
}
