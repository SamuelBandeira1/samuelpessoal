"use client";

import { DndContext, DragEndEvent, PointerSensor, useDroppable, useSensor, useSensors } from "@dnd-kit/core";
import { restrictToParentElement } from "@dnd-kit/modifiers";
import { SortableContext, verticalListSortingStrategy, useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { motion } from "framer-motion";
import { type CSSProperties, useMemo, useState } from "react";

import { RoleGuard } from "./role-guard";

interface Appointment {
  id: string;
  patient: string;
  provider: string;
  service: string;
  start: string;
  duration: number;
  status: "confirmado" | "pendente" | "retorno";
}

type DayKey = "monday" | "tuesday" | "wednesday" | "thursday" | "friday";

type ScheduleState = Record<DayKey, Appointment[]>;

const weekDays: Array<{ key: DayKey; label: string; date: string }> = [
  { key: "monday", label: "Seg", date: "15/04" },
  { key: "tuesday", label: "Ter", date: "16/04" },
  { key: "wednesday", label: "Qua", date: "17/04" },
  { key: "thursday", label: "Qui", date: "18/04" },
  { key: "friday", label: "Sex", date: "19/04" }
];

const providers = [
  { id: "sofia", name: "Dra. Sofia Lima" },
  { id: "caio", name: "Dr. Caio Moura" },
  { id: "aline", name: "Dra. Aline Costa" }
];

const services = [
  { id: "avaliacao", name: "Avaliação" },
  { id: "limpeza", name: "Limpeza" },
  { id: "lente", name: "Lente de contato" },
  { id: "clareamento", name: "Clareamento" }
];

const initialSchedule: ScheduleState = {
  monday: [
    {
      id: "a-1",
      patient: "João Nogueira",
      provider: "sofia",
      service: "avaliacao",
      start: "09:00",
      duration: 60,
      status: "confirmado"
    },
    {
      id: "a-2",
      patient: "Ana Paula",
      provider: "caio",
      service: "limpeza",
      start: "10:30",
      duration: 50,
      status: "pendente"
    }
  ],
  tuesday: [
    {
      id: "a-3",
      patient: "Carla Ribeiro",
      provider: "aline",
      service: "lente",
      start: "09:30",
      duration: 90,
      status: "confirmado"
    },
    {
      id: "a-4",
      patient: "Marcos Dias",
      provider: "sofia",
      service: "avaliacao",
      start: "15:00",
      duration: 45,
      status: "retorno"
    }
  ],
  wednesday: [
    {
      id: "a-5",
      patient: "Vitória Martins",
      provider: "caio",
      service: "clareamento",
      start: "11:00",
      duration: 80,
      status: "pendente"
    }
  ],
  thursday: [
    {
      id: "a-6",
      patient: "Lucas Ferreira",
      provider: "aline",
      service: "limpeza",
      start: "14:00",
      duration: 50,
      status: "confirmado"
    },
    {
      id: "a-7",
      patient: "Paula Rocha",
      provider: "caio",
      service: "avaliacao",
      start: "16:30",
      duration: 40,
      status: "retorno"
    }
  ],
  friday: [
    {
      id: "a-8",
      patient: "Eduardo Silva",
      provider: "sofia",
      service: "clareamento",
      start: "09:00",
      duration: 100,
      status: "confirmado"
    }
  ]
};

const statusStyles: Record<Appointment["status"], string> = {
  confirmado: "bg-emerald-100 text-emerald-700 border-emerald-200",
  pendente: "bg-amber-100 text-amber-700 border-amber-200",
  retorno: "bg-indigo-100 text-indigo-700 border-indigo-200"
};

export function ScheduleBoard() {
  const [schedule, setSchedule] = useState<ScheduleState>(initialSchedule);
  const [providerFilter, setProviderFilter] = useState<string>("all");
  const [serviceFilter, setServiceFilter] = useState<string>("all");

  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 6 } }));

  const filteredSchedule = useMemo(() => {
    if (providerFilter === "all" && serviceFilter === "all") {
      return schedule;
    }

    const filteredEntries = Object.entries(schedule).map(([day, items]) => [
      day,
      items.filter((appointment) => {
        const matchesProvider = providerFilter === "all" || appointment.provider === providerFilter;
        const matchesService = serviceFilter === "all" || appointment.service === serviceFilter;
        return matchesProvider && matchesService;
      })
    ]);

    return Object.fromEntries(filteredEntries) as ScheduleState;
  }, [providerFilter, schedule, serviceFilter]);

  const findDayByAppointment = (id: string): DayKey | undefined => {
    return weekDays.find(({ key }) => schedule[key].some((item) => item.id === id))?.key;
  };

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    if (!over) return;

    const fromDay = findDayByAppointment(active.id as string);
    if (!fromDay) return;

    let toDay = weekDays.find((day) => day.key === over.id)?.key;
    if (!toDay) {
      toDay = findDayByAppointment(over.id as string);
    }
    if (!toDay) return;

    setSchedule((prev) => {
      const sourceItems = [...prev[fromDay]];
      const movingIndex = sourceItems.findIndex((item) => item.id === active.id);
      if (movingIndex < 0) return prev;
      const [moving] = sourceItems.splice(movingIndex, 1);
      const nextState: ScheduleState = { ...prev, [fromDay]: sourceItems };

      const destinationItems = [...nextState[toDay]];
      const overIndex = destinationItems.findIndex((item) => item.id === over.id);
      if (overIndex >= 0) {
        destinationItems.splice(overIndex, 0, moving);
      } else {
        destinationItems.push(moving);
      }
      nextState[toDay] = destinationItems;
      return nextState;
    });
  };

  return (
    <section className="space-y-6">
      <header className="flex flex-col justify-between gap-4 md:flex-row md:items-end">
        <div>
          <h1 className="text-2xl font-semibold text-slate-900">Agenda semanal</h1>
          <p className="text-sm text-slate-500">Arraste para reagendar e mantenha a equipe sincronizada em tempo real.</p>
        </div>
        <div className="flex flex-wrap gap-3 text-sm">
          <label className="flex items-center gap-2">
            Profissional
            <select
              value={providerFilter}
              onChange={(event) => setProviderFilter(event.target.value)}
              className="rounded-full border border-slate-300 bg-white px-3 py-2 text-xs font-medium text-slate-600 shadow-sm focus:border-brand"
            >
              <option value="all">Todos</option>
              {providers.map((provider) => (
                <option key={provider.id} value={provider.id}>
                  {provider.name}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2">
            Procedimento
            <select
              value={serviceFilter}
              onChange={(event) => setServiceFilter(event.target.value)}
              className="rounded-full border border-slate-300 bg-white px-3 py-2 text-xs font-medium text-slate-600 shadow-sm focus:border-brand"
            >
              <option value="all">Todos</option>
              {services.map((service) => (
                <option key={service.id} value={service.id}>
                  {service.name}
                </option>
              ))}
            </select>
          </label>
        </div>
      </header>
      <DndContext sensors={sensors} onDragEnd={handleDragEnd} modifiers={[restrictToParentElement]}>
        <div className="grid gap-4 md:grid-cols-5">
          {weekDays.map((day) => (
            <DayColumn key={day.key} day={day} appointments={filteredSchedule[day.key]} />
          ))}
        </div>
      </DndContext>
      <RoleGuard
        allowed={["owner", "manager", "reception"]}
        fallback={
          <p className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-700">
            Dentistas conseguem apenas visualizar a agenda. Para reagendar, altere o perfil para recepção.
          </p>
        }
      >
        <p className="text-xs text-slate-400">
          Toda movimentação gera rastreabilidade automática em tempo real e notifica pacientes conforme preferências de canal.
        </p>
      </RoleGuard>
    </section>
  );
}

function DayColumn({
  day,
  appointments
}: {
  day: (typeof weekDays)[number];
  appointments: Appointment[];
}) {
  const { setNodeRef, isOver } = useDroppable({ id: day.key });

  return (
    <SortableContext items={appointments.map((appointment) => appointment.id)} strategy={verticalListSortingStrategy}>
      <div
        ref={setNodeRef}
        className={`flex min-h-[240px] flex-col gap-3 rounded-3xl border border-slate-200 bg-white/60 p-4 backdrop-blur transition ${
          isOver ? "ring-2 ring-brand/40" : ""
        }`}
        data-day={day.key}
      >
        <div className="flex items-baseline justify-between text-sm text-slate-500">
          <span className="font-semibold text-slate-700">{day.label}</span>
          <span>{day.date}</span>
        </div>
        {appointments.map((appointment) => (
          <ScheduleCard key={appointment.id} appointment={appointment} />
        ))}
        {appointments.length === 0 && (
          <p className="rounded-2xl border border-dashed border-slate-200 bg-white px-3 py-6 text-center text-xs text-slate-400">
            Sem consultas — arraste um paciente para cá
          </p>
        )}
      </div>
    </SortableContext>
  );
}

function ScheduleCard({ appointment }: { appointment: Appointment }) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({
    id: appointment.id
  });

  const style: CSSProperties = {
    transform: CSS.Transform.toString(transform),
    transition,
    zIndex: isDragging ? 10 : "auto"
  };

  const provider = providers.find((item) => item.id === appointment.provider)?.name ?? appointment.provider;
  const service = services.find((item) => item.id === appointment.service)?.name ?? appointment.service;

  return (
    <motion.article
      ref={setNodeRef}
      style={style}
      {...attributes}
      {...listeners}
      whileHover={{ scale: 1.01 }}
      whileTap={{ scale: 0.99 }}
      className={`cursor-grab rounded-2xl border p-4 shadow-sm ${statusStyles[appointment.status]}`}
    >
      <header className="flex items-start justify-between text-xs uppercase tracking-wide">
        <span>{appointment.start}</span>
        <span>{appointment.duration}min</span>
      </header>
      <h3 className="mt-2 text-sm font-semibold">{appointment.patient}</h3>
      <p className="mt-1 text-xs text-slate-600">{service}</p>
      <p className="mt-1 text-xs text-slate-500">{provider}</p>
      <span className="mt-3 inline-flex items-center rounded-full bg-white/70 px-3 py-1 text-[11px] font-semibold text-slate-700">
        {appointment.status === "confirmado" && "Confirmado"}
        {appointment.status === "pendente" && "Aguardando"}
        {appointment.status === "retorno" && "Retorno"}
      </span>
    </motion.article>
  );
}
