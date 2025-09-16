"use client";

import type { Role } from "./auth-context";
import { useAuth } from "./auth-context";

interface RoleGuardProps {
  allowed: Role[];
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

export function RoleGuard({ allowed, children, fallback }: RoleGuardProps) {
  const { user } = useAuth();

  if (!allowed.includes(user.role)) {
    return fallback ?? null;
  }

  return <>{children}</>;
}
