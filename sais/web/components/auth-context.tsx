"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

type Role = "owner" | "manager" | "reception" | "dentist";

type User = {
  name: string;
  role: Role;
  permissions: string[];
};

interface AuthContextValue {
  user: User;
  token: string;
  setRole: (role: Role) => void;
}

const STORAGE_KEY = "sais.dev.jwt";

const ROLE_CLAIMS: Record<Role, string[]> = {
  owner: ["view_metrics", "edit_services", "manage_templates", "manage_team"],
  manager: ["view_metrics", "edit_services", "manage_templates"],
  reception: ["manage_schedule", "manage_templates"],
  dentist: ["view_schedule", "update_records"]
};

const DEFAULT_USER: User = {
  name: "Clara Andrade",
  role: "owner",
  permissions: ROLE_CLAIMS.owner
};

const AuthContext = createContext<AuthContextValue | null>(null);

const encode = (value: string) => {
  if (typeof window !== "undefined" && window.btoa) {
    return window.btoa(value);
  }
  if (typeof Buffer !== "undefined") {
    return Buffer.from(value).toString("base64");
  }
  return value;
};

const decode = (value: string) => {
  if (typeof window !== "undefined" && window.atob) {
    return window.atob(value);
  }
  if (typeof Buffer !== "undefined") {
    return Buffer.from(value, "base64").toString("utf-8");
  }
  return value;
};

const buildUser = (role: Role, name?: string): User => ({
  name: name ?? DEFAULT_USER.name,
  role,
  permissions: ROLE_CLAIMS[role]
});

const decodeToken = (token: string | null): User | null => {
  if (!token) {
    return null;
  }
  try {
    const payload = JSON.parse(decode(token));
    const role = payload.role as Role | undefined;
    if (!role || !(role in ROLE_CLAIMS)) {
      return null;
    }
    const name = typeof payload.name === "string" ? payload.name : DEFAULT_USER.name;
    return buildUser(role, name);
  } catch (error) {
    console.warn("Falha ao decodificar token de desenvolvimento", error);
    return null;
  }
};

const encodeToken = (user: User) => encode(JSON.stringify({
  name: user.name,
  role: user.role,
  permissions: user.permissions
}));

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User>(DEFAULT_USER);
  const [token, setToken] = useState<string>(() => encodeToken(DEFAULT_USER));

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const persisted = window.localStorage.getItem(STORAGE_KEY);
    const decoded = decodeToken(persisted);
    if (decoded) {
      setUser(decoded);
      setToken(encodeToken(decoded));
    }
  }, []);

  const setRole = useCallback((role: Role) => {
    setUser((prev) => {
      const nextUser = buildUser(role, prev.name);
      setToken(encodeToken(nextUser));
      return nextUser;
    });
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.setItem(STORAGE_KEY, token);
  }, [token]);

  const value = useMemo<AuthContextValue>(() => ({
    user,
    token,
    setRole
  }), [setRole, token, user]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth deve ser utilizado dentro de AuthProvider");
  }
  return ctx;
}

export type { Role, User };
