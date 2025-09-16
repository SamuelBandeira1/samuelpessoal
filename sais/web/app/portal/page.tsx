export default function PortalPage() {
  return (
    <main className="mx-auto flex min-h-screen max-w-4xl flex-col gap-8 px-6 py-16">
      <header className="flex flex-col gap-2 border-b border-slate-200 pb-6">
        <p className="text-sm uppercase tracking-[0.2em] text-brand">Portal do paciente</p>
        <h1 className="text-4xl font-semibold text-slate-900">Acompanhe seus agendamentos</h1>
        <p className="text-slate-600">
          Consulte horários confirmados, solicite remarcações e converse com a equipe de atendimento em um só lugar.
        </p>
      </header>

      <section className="grid gap-6 md:grid-cols-2">
        <article className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <h2 className="text-xl font-semibold text-slate-900">Próxima consulta</h2>
          <p className="mt-3 text-sm text-slate-500">Nenhuma consulta agendada. Agende agora para garantir seu horário!</p>
          <button className="mt-4 w-full rounded-full bg-brand px-4 py-2 text-sm font-semibold text-white shadow hover:bg-brand-dark">
            Novo agendamento
          </button>
        </article>
        <article className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <h2 className="text-xl font-semibold text-slate-900">Comunicação</h2>
          <p className="mt-3 text-sm text-slate-500">
            Receba atualizações instantâneas sobre confirmações, instruções pré-consulta e mensagens da equipe.
          </p>
          <div className="mt-4 space-y-2 text-sm text-slate-600">
            <p>• Ative notificações pelo WhatsApp ou SMS.</p>
            <p>• Consulte o histórico de mensagens enviadas.</p>
          </div>
        </article>
      </section>
    </main>
  );
}
