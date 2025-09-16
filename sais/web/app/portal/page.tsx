"use client";

import { motion } from "framer-motion";

import { RoleGuard } from "../../components/role-guard";

const metrics = [
  {
    title: "Ocupação",
    value: "82%",
    delta: "+6%",
    description: "vs. semana anterior",
    roles: ["owner", "manager", "reception"],
    tone: "positive"
  },
  {
    title: "No-show",
    value: "4,5%",
    delta: "-2,1%",
    description: "com lembretes automáticos",
    roles: ["owner", "manager", "reception"],
    tone: "positive"
  },
  {
    title: "Satisfação (NPS)",
    value: "92",
    delta: "+12",
    description: "respostas pós-consulta",
    roles: ["owner", "manager"],
    tone: "positive"
  }
] as const;

const timeline = [
  { time: "08:30", patient: "Juliana Castro", service: "Avaliação", status: "Confirmado" },
  { time: "10:00", patient: "Rodrigo Lopes", service: "Limpeza", status: "Aguardando confirmação" },
  { time: "13:30", patient: "Bianca Prado", service: "Clareamento", status: "Check-in realizado" },
  { time: "16:00", patient: "Felipe Neves", service: "Manutenção ortodôntica", status: "Lembrete enviado" }
];

const automationHighlights = [
  {
    title: "Fluxos ativos",
    detail: "9 campanhas",
    caption: "Lembretes, pós-consulta e reativações",
    tone: "active"
  },
  {
    title: "Mensagens enviadas",
    detail: "1.284",
    caption: "Últimos 30 dias",
    tone: "neutral"
  },
  {
    title: "Conversões WhatsApp",
    detail: "37%",
    caption: "Agendamentos confirmados via bot",
    tone: "positive"
  }
];

export default function PortalPage() {
  return (
    <div className="space-y-8">
      <motion.header
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="flex flex-col gap-2"
      >
        <span className="text-xs uppercase tracking-[0.32em] text-brand">Visão geral</span>
        <h1 className="text-3xl font-semibold text-slate-900">Como está a operação hoje?</h1>
        <p className="text-sm text-slate-500">
          Analise indicadores em tempo real, antecipe picos de atendimento e garanta que a equipe esteja alinhada com a agenda
          inteligente.
        </p>
      </motion.header>

      <section className="grid gap-4 md:grid-cols-3">
        {metrics.map((metric, index) => (
          <RoleGuard
            key={metric.title}
            allowed={metric.roles}
            fallback={
              <motion.div
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 * index }}
                className="rounded-3xl border border-dashed border-slate-200 bg-white/60 p-6 text-sm text-slate-400"
              >
                Métrica disponível somente para gestão.
              </motion.div>
            }
          >
            <motion.article
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 * index }}
              className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm"
            >
              <header className="flex items-center justify-between text-xs uppercase text-slate-500">
                <span>{metric.title}</span>
                <span
                  className={`rounded-full px-3 py-1 text-[11px] font-semibold ${
                    metric.tone === "positive" ? "bg-emerald-100 text-emerald-700" : "bg-slate-100 text-slate-600"
                  }`}
                >
                  {metric.delta}
                </span>
              </header>
              <p className="mt-4 text-3xl font-semibold text-slate-900">{metric.value}</p>
              <p className="mt-2 text-sm text-slate-500">{metric.description}</p>
            </motion.article>
          </RoleGuard>
        ))}
      </section>

      <motion.section
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
        className="grid gap-6 lg:grid-cols-[2fr_1fr]"
      >
        <article className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-semibold text-slate-900">Agenda do dia</h2>
              <p className="text-sm text-slate-500">Monitoramento em tempo real com confirmações automáticas.</p>
            </div>
            <span className="rounded-full bg-brand/10 px-3 py-1 text-xs font-semibold text-brand">12 consultas</span>
          </div>
          <ol className="mt-6 space-y-4 text-sm text-slate-600">
            {timeline.map((item) => (
              <li
                key={item.patient}
                className="flex items-start gap-4 rounded-2xl border border-slate-100 bg-slate-50/60 p-4"
              >
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-white text-sm font-semibold text-brand">
                  {item.time}
                </div>
                <div>
                  <p className="font-semibold text-slate-900">{item.patient}</p>
                  <p className="text-xs text-slate-500">{item.service}</p>
                  <p className="mt-1 inline-flex rounded-full bg-emerald-100 px-2 py-1 text-[11px] font-medium text-emerald-700">
                    {item.status}
                  </p>
                </div>
              </li>
            ))}
          </ol>
        </article>
        <aside className="space-y-4">
          {automationHighlights.map((highlight, index) => (
            <motion.div
              key={highlight.title}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.15 * index }}
              className={`rounded-3xl border p-5 shadow-sm ${
                highlight.tone === "positive"
                  ? "border-emerald-200 bg-emerald-50"
                  : highlight.tone === "active"
                  ? "border-brand/40 bg-brand/10"
                  : "border-slate-200 bg-white"
              }`}
            >
              <p className="text-xs uppercase tracking-wide text-slate-500">{highlight.title}</p>
              <p className="mt-2 text-2xl font-semibold text-slate-900">{highlight.detail}</p>
              <p className="mt-1 text-xs text-slate-600">{highlight.caption}</p>
            </motion.div>
          ))}
        </aside>
      </motion.section>

      <motion.section
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.3 }}
        className="grid gap-6 md:grid-cols-2"
      >
        <article className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
          <h2 className="text-xl font-semibold text-slate-900">Fila de mensagens</h2>
          <p className="mt-2 text-sm text-slate-500">
            Mensagens humanizadas disparadas automaticamente conforme o status da jornada do paciente.
          </p>
          <ul className="mt-4 space-y-3 text-sm text-slate-600">
            <li className="flex items-center justify-between rounded-2xl bg-slate-50 px-4 py-3">
              <span>WhatsApp · Confirmação D-1</span>
              <span className="text-xs font-semibold text-emerald-600">32 enviadas</span>
            </li>
            <li className="flex items-center justify-between rounded-2xl bg-slate-50 px-4 py-3">
              <span>SMS · Reforço D-0</span>
              <span className="text-xs font-semibold text-brand">9 agendadas</span>
            </li>
            <li className="flex items-center justify-between rounded-2xl bg-slate-50 px-4 py-3">
              <span>E-mail · Pós-consulta NPS</span>
              <span className="text-xs font-semibold text-slate-500">41 respostas</span>
            </li>
          </ul>
        </article>
        <article className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
          <h2 className="text-xl font-semibold text-slate-900">Pendências da equipe</h2>
          <ul className="mt-4 space-y-3 text-sm text-slate-600">
            <li className="flex items-center justify-between rounded-2xl bg-amber-50 px-4 py-3 text-amber-700">
              <span>Confirmar encaixe da paciente Larissa (18h)</span>
              <button className="text-xs font-semibold uppercase tracking-wide">Resolver</button>
            </li>
            <li className="flex items-center justify-between rounded-2xl bg-white px-4 py-3">
              <span>Enviar orientações pré-cirúrgicas para Daniel</span>
              <button className="text-xs font-semibold text-brand">Enviar agora</button>
            </li>
            <li className="flex items-center justify-between rounded-2xl bg-white px-4 py-3">
              <span>Registrar retorno de tratamento ortodôntico</span>
              <button className="text-xs font-semibold text-brand">Abrir ficha</button>
            </li>
          </ul>
        </article>
      </motion.section>
    </div>
  );
}
