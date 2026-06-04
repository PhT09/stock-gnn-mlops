import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatPercent(n: number, digits = 1) {
  return `${n.toFixed(digits)}%`
}

export function formatNumber(n: number) {
  return new Intl.NumberFormat('vi-VN').format(n)
}
