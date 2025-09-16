"use client";

import Link from "next/link";
import { motion } from "framer-motion";

const stats = [
  { label: "Pacientes atendidos", value: "+12k" },
  { label: "Profissionais", value: "350" },
  { label: "Taxa de no-show", value: "-32%" }
];

const benefits = [
  {
    title: "Menos faltas",
    description: "Lembretes multicanal com confirmação automática reduzem o no-show e liberam encaixes com segurança."
  },
  {
    title: "Agenda inteligente",
    description: "Aplicamos buffers, bloqueios e preferências por profissional para garantir a melhor ocupação da equipe."
  },
  {
    title: "Experiência humanizada",
    description: "Crie jornadas personalizadas com mensagens no tom da clínica e encante pacientes em cada interação."
  }
];

const steps = [
  {
    title: "Descoberta",
    description: "Configuramos serviços, políticas de agendamento e integrações em poucos dias."
  },
  {
    title: "Conexão",
    description: "Importe pacientes ou ative formulários e bots para alimentar a base automaticamente."
  },
  {
    title: "Engajamento",
    description: "Fluxos inteligentes enviam convites, lembretes e pós-consulta no canal ideal."
  },
  {
    title: "Análise",
    description: "Dashboards atualizam KPIs em tempo real para decisões orientadas por dados."
  }
];

const features = [
  {
    title: "Orquestração Omnicanal",
    description: "WhatsApp, SMS, e-mail e telefone atuando em sincronia com regras de 24h e templates aprovados."
  },
  {
    title: "Fila inteligente",
    description: "Encaixes automáticos com priorização por urgência e preferências do paciente."
  },
  {
    title: "Relatórios acionáveis",
    description: "KPIs de ocupação, no-show, receita e engajamento com exportação em um clique."
  },
  {
    title: "Portal do paciente",
    description: "Autoatendimento, confirmação em um toque e histórico sempre disponível."
  },
  {
    title: "Integrações",
    description: "API aberta para ERPs odontológicos, pagamentos e soluções de telemedicina."
  },
  {
    title: "Suporte estratégico",
    description: "Time especialista acompanha sua operação para capturar oportunidades."
  }
];

const plans = [
  {
    name: "Essencial",
    price: "R$ 799/mês",
    items: ["Até 3 cadeiras", "Templates e fluxos prontos", "Portal paciente"],
    highlight: false
  },
  {
    name: "Growth",
    price: "R$ 1.299/mês",
    items: ["Até 6 cadeiras", "WhatsApp oficial incluso", "Dashboard de ocupação"],
    highlight: true
  },
  {
    name: "Enterprise",
    price: "Sob consulta",
    items: ["Multiclínicas", "Suporte dedicado", "Integrações personalizadas"],
    highlight: false
  }
];

const testimonials = [
  {
    name: "Dra. Fernanda Azevedo",
    role: "CEO · Rede Sorrir",
    quote: "Reduzimos no-show em 41% no primeiro trimestre com lembretes omnichannel e confirmação pelo bot."
  },
  {
    name: "Marcelo Andrade",
    role: "Diretor Operacional · A1 Odonto",
    quote: "O time ganhou visibilidade do dia e consegue reagir rápido a encaixes sem planilhas paralelas."
  },
  {
    name: "Paula Meireles",
    role: "Coordenadora de Atendimento · Studio Odonto",
    quote: "A jornada pós-consulta elevou nosso NPS para 92 e impulsionou indicações orgânicas."
  }
];

const faqs = [
  {
    question: "Quanto tempo leva para ativar o SAIS?",
    answer: "Em até 7 dias configuramos serviços, profissionais, templates e conectamos seu número oficial."
  },
  {
    question: "Posso usar meu WhatsApp atual?",
    answer: "Sim, basta validar o número na Evolution e conectamos o bot de atendimento sem perder histórico."
  },
  {
    question: "Como funciona a cobrança?",
    answer: "Planos mensais com contratação flexível, sem fidelidade. Créditos de mensagens são ajustados ao volume."
  },
  {
    question: "É necessário treinamento?",
    answer: "Disponibilizamos onboarding remoto e materiais sob demanda. Sua equipe domina o portal em horas."
  }
];

export default function LandingPage() {
  return (
    <main className="flex min-h-screen flex-col items-center bg-gradient-to-b from-white via-slate-50 to-slate-100 text-slate-900">
      <section className="relative w-full overflow-hidden px-6 pt-24 pb-16 sm:px-12 lg:px-24">
        <div className="mx-auto grid max-w-6xl gap-16 lg:grid-cols-2 lg:items-center">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            className="space-y-8"
          >
            <span className="inline-flex items-center rounded-full bg-brand/10 px-4 py-1 text-sm font-medium text-brand">
              Plataforma omnichannel para saúde inteligente
            </span>
            <h1 className="text-4xl font-bold tracking-tight sm:text-5xl lg:text-6xl">
              Centralize agendamentos, mensagens e performance da sua clínica em um só lugar.
            </h1>
            <p className="text-lg text-slate-600">
              O SAIS conecta sua clínica a pacientes com fluxos automatizados, lembretes inteligentes e painéis de performance
              para reduzir faltas e aumentar a satisfação.
            </p>
            <div className="flex flex-wrap items-center gap-4">
              <motion.div whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.98 }}>
                <Link
                  href="/portal"
                  className="rounded-full bg-brand px-6 py-3 text-base font-semibold text-white shadow-lg shadow-brand/40 transition hover:bg-brand-dark"
                >
                  Acessar Portal do Paciente
                </Link>
              </motion.div>
              <motion.div whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.98 }}>
                <a
                  href="#planos"
                  className="rounded-full border border-brand px-6 py-3 text-base font-semibold text-brand transition hover:border-brand-dark hover:text-brand-dark"
                >
                  Ver planos
                </a>
              </motion.div>
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
                <h2 className="text-lg font-semibold">Jornada do paciente</h2>
                <p className="text-sm text-slate-500">
                  Monitoramento em tempo real com alertas automáticos, notificações multicanal e visão 360º do histórico do
                  paciente.
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

      <motion.section
        initial={{ opacity: 0, y: 32 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.3 }}
        transition={{ duration: 0.6 }}
        className="w-full bg-white py-16"
      >
        <div className="mx-auto max-w-5xl space-y-10 px-6 sm:px-12">
          <div className="text-center">
            <p className="text-xs uppercase tracking-[0.32em] text-brand">Benefícios</p>
            <h2 className="mt-2 text-3xl font-semibold text-slate-900">Resultados tangíveis em semanas</h2>
          </div>
          <div className="grid gap-6 md:grid-cols-3">
            {benefits.map((benefit) => (
              <motion.article
                key={benefit.title}
                whileHover={{ translateY: -6 }}
                className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm"
              >
                <h3 className="text-xl font-semibold text-slate-900">{benefit.title}</h3>
                <p className="mt-3 text-sm text-slate-600">{benefit.description}</p>
              </motion.article>
            ))}
          </div>
        </div>
      </motion.section>

      <motion.section
        initial={{ opacity: 0, y: 32 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.4 }}
        transition={{ duration: 0.6 }}
        className="w-full bg-slate-50 py-16"
      >
        <div className="mx-auto max-w-5xl space-y-8 px-6 sm:px-12">
          <div className="text-center">
            <p className="text-xs uppercase tracking-[0.32em] text-brand">Como funciona</p>
            <h2 className="mt-2 text-3xl font-semibold text-slate-900">Fluxo rápido da implantação</h2>
          </div>
          <div className="grid gap-6 md:grid-cols-4">
            {steps.map((step, index) => (
              <div key={step.title} className="space-y-3 rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
                <span className="inline-flex h-10 w-10 items-center justify-center rounded-full bg-brand/10 text-sm font-semibold text-brand">
                  {index + 1}
                </span>
                <h3 className="text-lg font-semibold text-slate-900">{step.title}</h3>
                <p className="text-sm text-slate-600">{step.description}</p>
              </div>
            ))}
          </div>
        </div>
      </motion.section>

      <motion.section
        initial={{ opacity: 0, y: 32 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.3 }}
        transition={{ duration: 0.6 }}
        className="w-full bg-white py-16"
      >
        <div className="mx-auto max-w-6xl space-y-8 px-6 sm:px-12">
          <div className="text-center">
            <p className="text-xs uppercase tracking-[0.32em] text-brand">Recursos</p>
            <h2 className="mt-2 text-3xl font-semibold text-slate-900">Tecnologia pensada para operações de saúde</h2>
          </div>
          <div className="grid gap-6 md:grid-cols-3">
            {features.map((feature) => (
              <motion.article
                key={feature.title}
                whileHover={{ translateY: -4 }}
                className="rounded-3xl border border-slate-200 bg-white p-6 text-left shadow-sm"
              >
                <h3 className="text-lg font-semibold text-slate-900">{feature.title}</h3>
                <p className="mt-3 text-sm text-slate-600">{feature.description}</p>
              </motion.article>
            ))}
          </div>
        </div>
      </motion.section>

      <motion.section
        id="planos"
        initial={{ opacity: 0, y: 32 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.3 }}
        transition={{ duration: 0.6 }}
        className="w-full bg-slate-50 py-16"
      >
        <div className="mx-auto max-w-6xl space-y-8 px-6 sm:px-12">
          <div className="text-center">
            <p className="text-xs uppercase tracking-[0.32em] text-brand">Planos</p>
            <h2 className="mt-2 text-3xl font-semibold text-slate-900">Escolha o formato ideal para sua operação</h2>
          </div>
          <div className="grid gap-6 md:grid-cols-3">
            {plans.map((plan) => (
              <motion.article
                key={plan.name}
                whileHover={{ translateY: -6 }}
                className={`rounded-3xl border p-6 text-center shadow-sm ${
                  plan.highlight ? "border-brand bg-white" : "border-slate-200 bg-white"
                }`}
              >
                <p className="text-xs uppercase tracking-[0.32em] text-brand">{plan.name}</p>
                <p className="mt-3 text-3xl font-semibold text-slate-900">{plan.price}</p>
                <ul className="mt-4 space-y-2 text-sm text-slate-600">
                  {plan.items.map((item) => (
                    <li key={item}>• {item}</li>
                  ))}
                </ul>
                <motion.a
                  whileHover={{ scale: 1.03 }}
                  whileTap={{ scale: 0.97 }}
                  href="https://wa.me/5585988887777?text=Quero%20conhecer%20o%20SAIS"
                  className={`mt-6 inline-flex w-full justify-center rounded-full px-5 py-2 text-sm font-semibold transition ${
                    plan.highlight ? "bg-brand text-white" : "border border-brand text-brand"
                  }`}
                >
                  Falar com especialista
                </motion.a>
              </motion.article>
            ))}
          </div>
        </div>
      </motion.section>

      <motion.section
        initial={{ opacity: 0, y: 32 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.3 }}
        transition={{ duration: 0.6 }}
        className="w-full bg-white py-16"
      >
        <div className="mx-auto max-w-5xl space-y-8 px-6 sm:px-12">
          <div className="text-center">
            <p className="text-xs uppercase tracking-[0.32em] text-brand">Depoimentos</p>
            <h2 className="mt-2 text-3xl font-semibold text-slate-900">Clínicas que já aceleram com o SAIS</h2>
          </div>
          <div className="grid gap-6 md:grid-cols-3">
            {testimonials.map((testimonial) => (
              <motion.blockquote
                key={testimonial.name}
                whileHover={{ translateY: -4 }}
                className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm"
              >
                <p className="text-sm text-slate-600">“{testimonial.quote}”</p>
                <footer className="mt-4 text-sm font-semibold text-slate-900">
                  {testimonial.name}
                  <span className="block text-xs font-normal text-slate-500">{testimonial.role}</span>
                </footer>
              </motion.blockquote>
            ))}
          </div>
        </div>
      </motion.section>

      <motion.section
        initial={{ opacity: 0, y: 32 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.3 }}
        transition={{ duration: 0.6 }}
        className="w-full bg-slate-50 py-16"
      >
        <div className="mx-auto max-w-4xl space-y-6 px-6 sm:px-12">
          <div className="text-center">
            <p className="text-xs uppercase tracking-[0.32em] text-brand">FAQ</p>
            <h2 className="mt-2 text-3xl font-semibold text-slate-900">Perguntas frequentes</h2>
          </div>
          <div className="space-y-4">
            {faqs.map((faq) => (
              <details key={faq.question} className="rounded-3xl border border-slate-200 bg-white p-6 text-left shadow-sm">
                <summary className="cursor-pointer text-lg font-semibold text-slate-900">
                  {faq.question}
                </summary>
                <p className="mt-3 text-sm text-slate-600">{faq.answer}</p>
              </details>
            ))}
          </div>
        </div>
      </motion.section>

      <motion.section
        initial={{ opacity: 0, y: 32 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, amount: 0.3 }}
        transition={{ duration: 0.6 }}
        className="w-full bg-white py-20"
      >
        <div className="mx-auto flex max-w-3xl flex-col items-center gap-6 px-6 text-center sm:px-12">
          <h2 className="text-3xl font-semibold text-slate-900">Pronto para transformar a jornada dos pacientes?</h2>
          <p className="text-sm text-slate-600">
            Fale com nosso time pelo WhatsApp e descubra como o SAIS pode personalizar a operação para sua clínica.
          </p>
          <motion.a
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            href="https://wa.me/5585988887777?text=Quero%20conhecer%20o%20SAIS"
            className="inline-flex items-center gap-2 rounded-full bg-emerald-500 px-8 py-3 text-base font-semibold text-white shadow-lg shadow-emerald-500/40"
          >
            <span>Enviar mensagem no WhatsApp</span>
            <span aria-hidden>→</span>
          </motion.a>
        </div>
      </motion.section>
    </main>
  );
}
