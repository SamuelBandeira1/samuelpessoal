"use client";

import { FormEvent, useMemo, useState } from "react";
import { motion } from "framer-motion";

import { RoleGuard } from "../../../components/role-guard";

type Template = {
  id: string;
  name: string;
  channel: "WhatsApp" | "SMS" | "E-mail";
  category: string;
  body: string;
  active: boolean;
};

const initialTemplates: Template[] = [
  {
    id: "t-1",
    name: "Confirmação D-1",
    channel: "WhatsApp",
    category: "Lembrete",
    body: "Olá {{nome}}, confirmamos sua consulta amanhã às {{horario}}. Responda 1 para confirmar ou 2 para remarcar.",
    active: true
  },
  {
    id: "t-2",
    name: "Pós-consulta NPS",
    channel: "WhatsApp",
    category: "Pesquisa",
    body: "Obrigado por confiar na clínica, {{nome}}! De 0 a 10, qual a chance de indicar nossos serviços?",
    active: true
  },
  {
    id: "t-3",
    name: "Reativação",
    channel: "E-mail",
    category: "Follow-up",
    body: "Sentimos sua falta, {{nome}}! Agende uma avaliação de revisão com condições especiais este mês.",
    active: false
  }
];

const channelOptions: Template["channel"][] = ["WhatsApp", "SMS", "E-mail"];

export default function TemplatesPage() {
  return (
    <RoleGuard
      allowed={["owner", "manager", "reception"]}
      fallback={
        <div className="rounded-3xl border border-amber-200 bg-amber-50 p-6 text-sm text-amber-700">
          Apenas a recepção e gestão podem editar templates transacionais.
        </div>
      }
    >
      <TemplatesManager />
    </RoleGuard>
  );
}

function TemplatesManager() {
  const [templates, setTemplates] = useState<Template[]>(initialTemplates);
  const [editing, setEditing] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [form, setForm] = useState({ name: "", channel: channelOptions[0], category: "Lembrete", body: "" });

  const filteredTemplates = useMemo(() => {
    if (!search) return templates;
    const query = search.toLowerCase();
    return templates.filter((template) =>
      [template.name, template.body, template.category, template.channel].some((field) =>
        field.toLowerCase().includes(query)
      )
    );
  }, [search, templates]);

  const reset = () => {
    setForm({ name: "", channel: channelOptions[0], category: "Lembrete", body: "" });
    setEditing(null);
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const payload: Template = {
      id: editing ?? `t-${Date.now()}`,
      name: form.name,
      channel: form.channel,
      category: form.category,
      body: form.body,
      active: editing ? templates.find((item) => item.id === editing)?.active ?? true : true
    };

    if (editing) {
      setTemplates((prev) => prev.map((template) => (template.id === editing ? payload : template)));
    } else {
      setTemplates((prev) => [payload, ...prev]);
    }

    reset();
  };

  const startEdit = (template: Template) => {
    setEditing(template.id);
    setForm({ name: template.name, channel: template.channel, category: template.category, body: template.body });
  };

  const toggleActive = (id: string) => {
    setTemplates((prev) =>
      prev.map((template) => (template.id === id ? { ...template, active: !template.active } : template))
    );
  };

  const removeTemplate = (id: string) => {
    setTemplates((prev) => prev.filter((template) => template.id !== id));
    if (editing === id) {
      reset();
    }
  };

  const preview = form.body
    ? form.body.replace(/{{nome}}/gi, "Mariana").replace(/{{horario}}/gi, "15h30")
    : "Selecione um template ou preencha o conteúdo para visualizar a prévia";

  return (
    <div className="space-y-8">
      <header className="flex flex-col gap-2">
        <span className="text-xs uppercase tracking-[0.32em] text-brand">Templates</span>
        <h1 className="text-3xl font-semibold text-slate-900">Personalize mensagens inteligentes</h1>
        <p className="text-sm text-slate-500">
          Ajuste conteúdos e canais de relacionamento para manter uma experiência consistente com a marca.
        </p>
      </header>

      <motion.form
        onSubmit={handleSubmit}
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm"
      >
        <div className="grid gap-4 md:grid-cols-4">
          <label className="md:col-span-2 text-sm font-medium text-slate-600">
            Nome interno
            <input
              required
              value={form.name}
              onChange={(event) => setForm((prev) => ({ ...prev, name: event.target.value }))}
              className="mt-2 w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm focus:border-brand focus:outline-none"
            />
          </label>
          <label className="text-sm font-medium text-slate-600">
            Canal
            <select
              value={form.channel}
              onChange={(event) => setForm((prev) => ({ ...prev, channel: event.target.value as Template["channel"] }))}
              className="mt-2 w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm focus:border-brand focus:outline-none"
            >
              {channelOptions.map((channel) => (
                <option key={channel}>{channel}</option>
              ))}
            </select>
          </label>
          <label className="text-sm font-medium text-slate-600">
            Categoria
            <input
              value={form.category}
              onChange={(event) => setForm((prev) => ({ ...prev, category: event.target.value }))}
              className="mt-2 w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm focus:border-brand focus:outline-none"
            />
          </label>
          <label className="md:col-span-4 text-sm font-medium text-slate-600">
            Conteúdo
            <textarea
              required
              value={form.body}
              onChange={(event) => setForm((prev) => ({ ...prev, body: event.target.value }))}
              rows={4}
              placeholder="Use variáveis como {{nome}} e {{horario}}"
              className="mt-2 w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm focus:border-brand focus:outline-none"
            />
          </label>
        </div>
        <div className="mt-4 flex flex-wrap items-center gap-3">
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            type="submit"
            className="rounded-full bg-brand px-5 py-2 text-sm font-semibold text-white shadow-brand/40"
          >
            {editing ? "Atualizar template" : "Adicionar template"}
          </motion.button>
          {editing && (
            <button type="button" onClick={reset} className="text-sm font-medium text-slate-500 underline">
              Cancelar edição
            </button>
          )}
        </div>
      </motion.form>

      <section className="grid gap-6 lg:grid-cols-[2fr_1fr]">
        <div className="space-y-3">
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <h2 className="text-lg font-semibold text-slate-900">Biblioteca ativa</h2>
            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Buscar templates"
              className="w-full rounded-full border border-slate-200 bg-white px-4 py-2 text-sm focus:border-brand focus:outline-none md:w-72"
            />
          </div>
          <div className="space-y-3">
            {filteredTemplates.map((template) => (
              <motion.article
                key={template.id}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                className="rounded-3xl border border-slate-200 bg-white/80 p-5 shadow-sm"
              >
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <p className="text-base font-semibold text-slate-900">{template.name}</p>
                    <p className="text-xs uppercase tracking-wide text-slate-400">
                      {template.channel} · {template.category}
                    </p>
                  </div>
                  <div className="flex gap-2 text-xs font-semibold">
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => toggleActive(template.id)}
                      className={`rounded-full px-4 py-2 ${
                        template.active
                          ? "bg-emerald-100 text-emerald-700"
                          : "border border-slate-200 text-slate-500"
                      }`}
                    >
                      {template.active ? "Ativo" : "Inativo"}
                    </motion.button>
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => startEdit(template)}
                      className="rounded-full border border-brand px-4 py-2 text-brand"
                    >
                      Editar
                    </motion.button>
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => removeTemplate(template.id)}
                      className="rounded-full border border-red-200 px-4 py-2 text-red-500"
                    >
                      Excluir
                    </motion.button>
                  </div>
                </div>
                <p className="mt-3 text-sm text-slate-600">{template.body}</p>
              </motion.article>
            ))}
            {filteredTemplates.length === 0 && (
              <p className="rounded-3xl border border-dashed border-slate-200 bg-white py-8 text-center text-sm text-slate-400">
                Nenhum template corresponde a esse filtro.
              </p>
            )}
          </div>
        </div>
        <aside className="h-full rounded-3xl border border-slate-200 bg-slate-900/95 p-6 text-white shadow-lg">
          <h3 className="text-sm uppercase tracking-[0.3em] text-slate-400">Prévia</h3>
          <p className="mt-4 whitespace-pre-wrap text-sm leading-relaxed text-slate-100">{preview}</p>
          <p className="mt-6 text-xs text-slate-400">
            Variáveis suportadas: {"{{nome}}"}, {"{{horario}}"}, {"{{servico}}"}, {"{{link_confirmacao}}"}.
          </p>
        </aside>
      </section>
    </div>
  );
}
