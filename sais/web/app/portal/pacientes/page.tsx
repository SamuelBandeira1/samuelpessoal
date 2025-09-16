"use client";

import { FormEvent, useMemo, useState } from "react";
import { motion } from "framer-motion";

import { RoleGuard } from "../../../components/role-guard";

type Patient = {
  id: string;
  name: string;
  phone: string;
  lastVisit: string;
  tags: string[];
  notes?: string;
};

const initialPatients: Patient[] = [
  {
    id: "p-1",
    name: "Larissa Monteiro",
    phone: "+55 85 98888-1234",
    lastVisit: "12/03/2024",
    tags: ["Ortodontia", "VIP"],
    notes: "Prefere contato por WhatsApp."
  },
  {
    id: "p-2",
    name: "Daniel Furtado",
    phone: "+55 85 99777-4321",
    lastVisit: "04/04/2024",
    tags: ["Implante"],
    notes: "Enviar lembrete com instruções pré-cirúrgicas."
  }
];

export default function PatientsPage() {
  return (
    <RoleGuard
      allowed={["owner", "manager", "reception"]}
      fallback={
        <div className="rounded-3xl border border-amber-200 bg-amber-50 p-6 text-sm text-amber-700">
          Acesso restrito: somente a recepção e gestão podem editar cadastros de pacientes.
        </div>
      }
    >
      <PatientsManager />
    </RoleGuard>
  );
}

function PatientsManager() {
  const [patients, setPatients] = useState<Patient[]>(initialPatients);
  const [editing, setEditing] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [form, setForm] = useState({ name: "", phone: "", tags: "", notes: "" });

  const filteredPatients = useMemo(() => {
    if (!search) return patients;
    const query = search.toLowerCase();
    return patients.filter((patient) =>
      [patient.name, patient.phone, patient.tags.join(","), patient.notes ?? ""].some((field) =>
        field.toLowerCase().includes(query)
      )
    );
  }, [patients, search]);

  const resetForm = () => {
    setForm({ name: "", phone: "", tags: "", notes: "" });
    setEditing(null);
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const payload: Patient = {
      id: editing ?? `p-${Date.now()}`,
      name: form.name,
      phone: form.phone,
      lastVisit: editing ? patients.find((item) => item.id === editing)?.lastVisit ?? "--" : "--",
      tags: form.tags
        .split(",")
        .map((tag) => tag.trim())
        .filter(Boolean),
      notes: form.notes
    };

    if (editing) {
      setPatients((prev) => prev.map((item) => (item.id === editing ? { ...payload } : item)));
    } else {
      setPatients((prev) => [{ ...payload, lastVisit: "--" }, ...prev]);
    }
    resetForm();
  };

  const startEdit = (patient: Patient) => {
    setEditing(patient.id);
    setForm({
      name: patient.name,
      phone: patient.phone,
      tags: patient.tags.join(", "),
      notes: patient.notes ?? ""
    });
  };

  const removePatient = (id: string) => {
    setPatients((prev) => prev.filter((patient) => patient.id !== id));
    if (editing === id) {
      resetForm();
    }
  };

  return (
    <div className="space-y-8">
      <header className="flex flex-col gap-2">
        <span className="text-xs uppercase tracking-[0.32em] text-brand">Pacientes</span>
        <h1 className="text-3xl font-semibold text-slate-900">Centralize cadastros e contatos</h1>
        <p className="text-sm text-slate-500">
          Cadastre novos pacientes, atualize preferências de comunicação e mantenha a equipe alinhada com históricos.
        </p>
      </header>

      <motion.form
        onSubmit={handleSubmit}
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm"
      >
        <div className="grid gap-4 md:grid-cols-2">
          <label className="text-sm font-medium text-slate-600">
            Nome completo
            <input
              required
              value={form.name}
              onChange={(event) => setForm((prev) => ({ ...prev, name: event.target.value }))}
              className="mt-2 w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm focus:border-brand focus:outline-none"
            />
          </label>
          <label className="text-sm font-medium text-slate-600">
            Telefone
            <input
              required
              value={form.phone}
              onChange={(event) => setForm((prev) => ({ ...prev, phone: event.target.value }))}
              placeholder="+55 85 9XXXX-XXXX"
              className="mt-2 w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm focus:border-brand focus:outline-none"
            />
          </label>
          <label className="text-sm font-medium text-slate-600">
            Tags
            <input
              value={form.tags}
              onChange={(event) => setForm((prev) => ({ ...prev, tags: event.target.value }))}
              placeholder="Implante, VIP"
              className="mt-2 w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm focus:border-brand focus:outline-none"
            />
          </label>
          <label className="text-sm font-medium text-slate-600 md:col-span-2">
            Observações
            <textarea
              value={form.notes}
              onChange={(event) => setForm((prev) => ({ ...prev, notes: event.target.value }))}
              rows={3}
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
            {editing ? "Atualizar paciente" : "Adicionar paciente"}
          </motion.button>
          {editing && (
            <button
              type="button"
              onClick={resetForm}
              className="text-sm font-medium text-slate-500 underline"
            >
              Cancelar edição
            </button>
          )}
        </div>
      </motion.form>

      <section className="space-y-4">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <h2 className="text-lg font-semibold text-slate-900">Pacientes ativos</h2>
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Buscar por nome, telefone ou tag"
            className="w-full rounded-full border border-slate-200 bg-white px-4 py-2 text-sm focus:border-brand focus:outline-none md:w-72"
          />
        </div>
        <div className="space-y-3">
          {filteredPatients.map((patient) => (
            <motion.article
              key={patient.id}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex flex-col gap-3 rounded-3xl border border-slate-200 bg-white/80 p-5 shadow-sm md:flex-row md:items-center md:justify-between"
            >
              <div>
                <p className="text-lg font-semibold text-slate-900">{patient.name}</p>
                <p className="text-sm text-slate-500">{patient.phone}</p>
                <p className="text-xs text-slate-400">Última visita: {patient.lastVisit}</p>
                <div className="mt-2 flex flex-wrap gap-2 text-xs text-brand">
                  {patient.tags.map((tag) => (
                    <span key={tag} className="rounded-full bg-brand/10 px-3 py-1 font-semibold">
                      {tag}
                    </span>
                  ))}
                </div>
                {patient.notes && <p className="mt-2 text-xs text-slate-500">{patient.notes}</p>}
              </div>
              <div className="flex gap-2 text-sm">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => startEdit(patient)}
                  className="rounded-full border border-brand px-4 py-2 font-semibold text-brand"
                >
                  Editar
                </motion.button>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => removePatient(patient.id)}
                  className="rounded-full border border-red-200 px-4 py-2 font-semibold text-red-500"
                >
                  Remover
                </motion.button>
              </div>
            </motion.article>
          ))}
          {filteredPatients.length === 0 && (
            <p className="rounded-3xl border border-dashed border-slate-200 bg-white py-8 text-center text-sm text-slate-400">
              Nenhum paciente encontrado com esse filtro.
            </p>
          )}
        </div>
      </section>
    </div>
  );
}
