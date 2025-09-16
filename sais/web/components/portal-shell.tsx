"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { motion } from "framer-motion";

import type { Role } from "./auth-context";
import { useAuth } from "./auth-context";
import { ConsentModal } from "./consent-modal";
import { RoleGuard } from "./role-guard";

const navigation: Array<{ label: string; href: string; roles: Role[] }> = [
  { label: "Dashboard", href: "/portal", roles: ["owner", "manager", "reception", "dentist"] },
  { label: "Agenda", href: "/portal/agenda", roles: ["owner", "manager", "reception", "dentist"] },
  { label: "Pacientes", href: "/portal/pacientes", roles: ["owner", "manager", "reception"] },
  { label: "Serviços", href: "/portal/servicos", roles: ["owner", "manager"] },
  { label: "Templates", href: "/portal/templates", roles: ["owner", "manager", "reception"] }
];

const roleOptions: Array<{ value: Role; label: string }> = [
  { value: "owner", label: "Owner" },
  { value: "manager", label: "Manager" },
  { value: "reception", label: "Reception" },
  { value: "dentist", label: "Dentist" }
];

export default function PortalShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { user, setRole, token } = useAuth();
  const [navOpen, setNavOpen] = useState(false);

  return (
    <div className="min-h-screen bg-slate-100">
      <ConsentModal />
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-6 py-4">
          <div className="flex items-center gap-3">
            <button
              type="button"
              className="rounded-full border border-slate-200 p-2 text-slate-500 transition hover:border-brand hover:text-brand lg:hidden"
              onClick={() => setNavOpen((prev) => !prev)}
              aria-label="Alternar navegação"
            >
              <span className="block h-0.5 w-4 bg-current" />
              <span className="mt-1 block h-0.5 w-4 bg-current" />
              <span className="mt-1 block h-0.5 w-4 bg-current" />
            </button>
            <div>
              <p className="text-xs uppercase tracking-[0.35em] text-brand">SAIS Portal</p>
              <h1 className="text-lg font-semibold text-slate-900">Bem-vinda, {user.name}</h1>
            </div>
          </div>
          <div className="flex flex-col items-end gap-1 text-xs text-slate-500">
            <RoleSwitcher selected={user.role} onChange={setRole} />
            <span className="max-w-[220px] truncate font-mono" title={token}>
              {token}
            </span>
          </div>
        </div>
      </header>
      <div className="mx-auto flex max-w-6xl flex-col gap-6 px-6 py-6 lg:flex-row">
        <motion.nav
          layout
          className={`overflow-hidden rounded-3xl border border-slate-200 bg-white/80 p-4 shadow-sm backdrop-blur transition-all lg:sticky lg:top-6 lg:max-h-[calc(100vh-3rem)] lg:w-64 ${
            navOpen ? "max-h-96" : "max-h-0 lg:max-h-full"
          }`}
        >
          <ul className="flex flex-col gap-2 text-sm">
            {navigation.map((item) => (
              <RoleGuard key={item.href} allowed={item.roles}>
                <li>
                  <Link
                    href={item.href}
                    className={`flex items-center justify-between rounded-2xl px-4 py-3 transition ${
                      pathname === item.href
                        ? "bg-brand text-white shadow-brand/30"
                        : "text-slate-600 hover:bg-slate-100"
                    }`}
                    onClick={() => setNavOpen(false)}
                  >
                    {item.label}
                    <span aria-hidden>›</span>
                  </Link>
                </li>
              </RoleGuard>
            ))}
          </ul>
        </motion.nav>
        <main className="flex-1 space-y-8 pb-12">{children}</main>
      </div>
    </div>
  );
}

function RoleSwitcher({ selected, onChange }: { selected: Role; onChange: (role: Role) => void }) {
  return (
    <label className="flex flex-col items-end gap-1 text-xs font-medium text-slate-500">
      Perfil ativo
      <motion.select
        whileFocus={{ scale: 1.02 }}
        className="rounded-full border border-slate-300 bg-white px-3 py-2 text-xs font-semibold text-slate-700 shadow-sm focus:border-brand focus:outline-none"
        value={selected}
        onChange={(event) => onChange(event.target.value as Role)}
      >
        {roleOptions.map((role) => (
          <option key={role.value} value={role.value}>
            {role.label}
          </option>
        ))}
      </motion.select>
    </label>
  );
}
