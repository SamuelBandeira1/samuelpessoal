"use client";

import { FormEvent, useMemo, useState } from "react";
import { motion } from "framer-motion";

import { RoleGuard } from "../../../components/role-guard";

type Service = {
  id: string;
  name: string;
  duration: number;
  price: number;
  buffer: number;
};

const initialServices: Service[] = [
  { id: "s-1", name: "Consulta de avaliação", duration: 40, price: 180, buffer: 10 },
  { id: "s-2", name: "Limpeza preventiva", duration: 50, price: 220, buffer: 5 },
  { id: "s-3", name: "Clareamento consultório", duration: 90, price: 680, buffer: 15 }
];

const currency = new Intl.NumberFormat("pt-BR", { style: "currency", currency: "BRL" });

export default function ServicesPage() {
  return (
    <RoleGuard
      allowed={["owner", "manager"]}
      fallback={
        <div className="rounded-3xl border border-amber-200 bg-amber-50 p-6 text-sm text-amber-700">
          Somente gestão pode editar os serviços ofertados. Consulte o responsável para solicitar alterações.
        </div>
      }
    >
      <ServicesManager />
    </RoleGuard>
  );
}

function ServicesManager() {
  const [services, setServices] = useState<Service[]>(initialServices);
  const [editing, setEditing] = useState<string | null>(null);
  const [form, setForm] = useState({ name: "", duration: 30, price: 150, buffer: 5 });
  const [search, setSearch] = useState("");

  const filteredServices = useMemo(() => {
    if (!search) return services;
    const query = search.toLowerCase();
    return services.filter((service) => service.name.toLowerCase().includes(query));
  }, [search, services]);

  const reset = () => {
    setForm({ name: "", duration: 30, price: 150, buffer: 5 });
    setEditing(null);
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const payload: Service = {
      id: editing ?? `s-${Date.now()}`,
      name: form.name,
      duration: Number(form.duration),
      price: Number(form.price),
      buffer: Number(form.buffer)
    };

    if (editing) {
      setServices((prev) => prev.map((service) => (service.id === editing ? payload : service)));
    } else {
      setServices((prev) => [payload, ...prev]);
    }

    reset();
  };

  const startEdit = (service: Service) => {
    setEditing(service.id);
    setForm({ name: service.name, duration: service.duration, price: service.price, buffer: service.buffer });
  };

  const removeService = (id: string) => {
    setServices((prev) => prev.filter((service) => service.id !== id));
    if (editing === id) {
      reset();
    }
  };

  return (
    <div className="space-y-8">
      <header className="flex flex-col gap-2">
        <span className="text-xs uppercase tracking-[0.32em] text-brand">Serviços</span>
        <h1 className="text-3xl font-semibold text-slate-900">Defina regras inteligentes por procedimento</h1>
        <p className="text-sm text-slate-500">
          Configure duração, buffers obrigatórios e precificação para otimizar a agenda com base em tipos de procedimentos.
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
            Nome do serviço
            <input
              required
              value={form.name}
              onChange={(event) => setForm((prev) => ({ ...prev, name: event.target.value }))}
              className="mt-2 w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm focus:border-brand focus:outline-none"
            />
          </label>
          <label className="text-sm font-medium text-slate-600">
            Duração (min)
            <input
              type="number"
              min={15}
              value={form.duration}
              onChange={(event) => setForm((prev) => ({ ...prev, duration: Number(event.target.value) }))}
              className="mt-2 w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm focus:border-brand focus:outline-none"
            />
          </label>
          <label className="text-sm font-medium text-slate-600">
            Buffer (min)
            <input
              type="number"
              min={0}
              value={form.buffer}
              onChange={(event) => setForm((prev) => ({ ...prev, buffer: Number(event.target.value) }))}
              className="mt-2 w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm focus:border-brand focus:outline-none"
            />
          </label>
          <label className="text-sm font-medium text-slate-600">
            Preço (R$)
            <input
              type="number"
              min={0}
              step={10}
              value={form.price}
              onChange={(event) => setForm((prev) => ({ ...prev, price: Number(event.target.value) }))}
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
            {editing ? "Atualizar serviço" : "Adicionar serviço"}
          </motion.button>
          {editing && (
            <button type="button" onClick={reset} className="text-sm font-medium text-slate-500 underline">
              Cancelar edição
            </button>
          )}
        </div>
      </motion.form>

      <section className="space-y-4">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <h2 className="text-lg font-semibold text-slate-900">Tabela de serviços</h2>
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Buscar serviço"
            className="w-full rounded-full border border-slate-200 bg-white px-4 py-2 text-sm focus:border-brand focus:outline-none md:w-64"
          />
        </div>
        <div className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
          <table className="min-w-full divide-y divide-slate-200 text-sm">
            <thead className="bg-slate-50 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-6 py-4">Serviço</th>
                <th className="px-6 py-4">Duração</th>
                <th className="px-6 py-4">Buffer</th>
                <th className="px-6 py-4">Preço</th>
                <th className="px-6 py-4 text-right">Ações</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 bg-white">
              {filteredServices.map((service) => (
                <tr key={service.id} className="transition hover:bg-slate-50/80">
                  <td className="px-6 py-4 font-semibold text-slate-900">{service.name}</td>
                  <td className="px-6 py-4 text-slate-500">{service.duration} min</td>
                  <td className="px-6 py-4 text-slate-500">{service.buffer} min</td>
                  <td className="px-6 py-4 text-slate-500">{currency.format(service.price)}</td>
                  <td className="px-6 py-4">
                    <div className="flex justify-end gap-2">
                      <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={() => startEdit(service)}
                        className="rounded-full border border-brand px-4 py-2 text-xs font-semibold text-brand"
                      >
                        Editar
                      </motion.button>
                      <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={() => removeService(service.id)}
                        className="rounded-full border border-red-200 px-4 py-2 text-xs font-semibold text-red-500"
                      >
                        Remover
                      </motion.button>
                    </div>
                  </td>
                </tr>
              ))}
              {filteredServices.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-6 py-10 text-center text-sm text-slate-400">
                    Nenhum serviço encontrado com esse termo.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
