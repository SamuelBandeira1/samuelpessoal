"use client";

import Link from "next/link";
import { motion } from "framer-motion";

const heroVariants = {
  hidden: { opacity: 0, y: 40 },
  visible: { opacity: 1, y: 0 }
};

const stats = [
  { label: "Pacientes atendidos", value: "+12k" },
  { label: "Profissionais", value: "350" },
  { label: "Taxa de no-show", value: "-32%" }
];

export default function LandingPage() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between bg-gradient-to-b from-white via-slate-50 to-slate-100">
      <section className="relative w-full overflow-hidden px-6 pt-24 pb-16 sm:px-12 lg:px-24">
        <div className="mx-auto grid max-w-6xl gap-16 lg:grid-cols-2 lg:items-center">
          <motion.div
            initial="hidden"
            animate="visible"
            transition={{ duration: 0.8, ease: "easeOut" }}
            variants={heroVariants}
            className="space-y-8"
          >
            <span className="inline-flex items-center rounded-full bg-brand/10 px-4 py-1 text-sm font-medium text-brand">
              Plataforma omnichannel para saúde inteligente
            </span>
            <h1 className="text-4xl font-bold tracking-tight text-slate-900 sm:text-5xl lg:text-6xl">
              Centralize agendamentos, mensagens e jornadas do paciente em um único lugar.
            </h1>
            <p className="text-lg text-slate-600">
              O SAIS conecta sua clínica a pacientes com fluxos automatizados, lembretes inteligentes e
              painéis de performance para reduzir faltas e aumentar a satisfação.
            </p>
            <div className="flex flex-wrap items-center gap-4">
              <Link
                href="/portal"
                className="rounded-full bg-brand px-6 py-3 text-base font-semibold text-white shadow-lg shadow-brand/40 transition hover:bg-brand-dark"
              >
                Acessar Portal do Paciente
              </Link>
              <a
                href="#recursos"
                className="rounded-full border border-brand px-6 py-3 text-base font-semibold text-brand transition hover:border-brand-dark hover:text-brand-dark"
              >
                Explorar recursos
              </a>
            </div>
            <dl className="grid gap-6 sm:grid-cols-3">
              {stats.map((stat) => (
                <div key={stat.label} className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
                  <dt className="text-sm font-medium text-slate-500">{stat.label}</dt>
                  <dd className="mt-2 text-3xl font-semibold text-slate-900">{stat.value}</dd>
                </div>
              ))}
            </dl>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 60 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.9, ease: "easeOut", delay: 0.2 }}
            className="relative"
          >
            <div className="relative mx-auto max-w-md overflow-hidden rounded-3xl bg-white p-6 shadow-xl ring-1 ring-slate-900/5">
              <div className="space-y-4">
                <h2 className="text-lg font-semibold text-slate-900">Jornada do paciente</h2>
                <p className="text-sm text-slate-500">
                  Monitoramento em tempo real com alertas automáticos, notificações multicanal e visão 360º do
                  histórico do paciente.
                </p>
                <ul className="space-y-3 text-sm">
                  <li className="flex items-start gap-2">
                    <span className="mt-1 h-2.5 w-2.5 rounded-full bg-emerald-400" />
                    <span>Lembretes com confirmação por WhatsApp, SMS e e-mail.</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="mt-1 h-2.5 w-2.5 rounded-full bg-sky-400" />
                    <span>Painel de ocupação com previsão baseada em machine learning.</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="mt-1 h-2.5 w-2.5 rounded-full bg-violet-400" />
                    <span>Check-in digital e automação de pós-consulta.</span>
                  </li>
                </ul>
              </div>
              <div className="mt-8 rounded-2xl bg-slate-900 px-6 py-5 text-white">
                <p className="text-sm uppercase tracking-wide text-slate-300">Taxa de confirmação</p>
                <p className="mt-2 text-4xl font-bold">92%</p>
                <p className="mt-1 text-sm text-slate-300">+18% vs. média do mercado</p>
              </div>
            </div>
            <div className="absolute inset-0 -z-10 rounded-full bg-brand/20 blur-3xl" aria-hidden />
          </motion.div>
        </div>
      </section>

      <section id="recursos" className="w-full bg-white py-16">
        <div className="mx-auto max-w-5xl space-y-8 px-6 sm:px-12">
          <h2 className="text-center text-3xl font-bold text-slate-900">Recursos essenciais</h2>
          <div className="grid gap-8 md:grid-cols-3">
            {["Orquestração Omnicanal", "Gestão de agendas", "Insights acionáveis"].map((feature) => (
              <div key={feature} className="rounded-2xl border border-slate-200 p-6 shadow-sm transition hover:-translate-y-1 hover:shadow-lg">
                <h3 className="text-xl font-semibold text-slate-900">{feature}</h3>
                <p className="mt-3 text-sm text-slate-600">
                  Automatize fluxos críticos do atendimento, personalize jornadas e acompanhe métricas em tempo real.
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
