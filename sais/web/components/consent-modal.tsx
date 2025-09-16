"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useState } from "react";

const STORAGE_KEY = "sais.lgpd.consent";

export function ConsentModal() {
  const [open, setOpen] = useState(false);
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const persisted = window.localStorage.getItem(STORAGE_KEY);
    if (!persisted) {
      const timer = window.setTimeout(() => setOpen(true), 600);
      return () => window.clearTimeout(timer);
    }
  }, []);

  const handleConfirm = () => {
    if (!checked) return;
    if (typeof window !== "undefined") {
      window.localStorage.setItem(STORAGE_KEY, "accepted");
    }
    setOpen(false);
  };

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 px-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          <motion.div
            className="max-w-lg rounded-3xl bg-white p-8 shadow-2xl"
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            transition={{ type: "spring", stiffness: 220, damping: 24 }}
          >
            <h2 className="text-2xl font-semibold text-slate-900">Consentimento LGPD</h2>
            <p className="mt-4 text-sm text-slate-600">
              Para enviar lembretes de consulta e confirmações automáticas, precisamos do seu consentimento. Os dados serão
              usados apenas para comunicação assistida e você pode revogar a qualquer momento.
            </p>
            <label className="mt-6 flex items-start gap-3 text-sm text-slate-700">
              <input
                type="checkbox"
                className="mt-1 h-4 w-4 rounded border-slate-300 text-brand focus:ring-brand"
                checked={checked}
                onChange={(event) => setChecked(event.target.checked)}
              />
              <span>
                Estou ciente e autorizo o envio de lembretes de consultas via WhatsApp, SMS e e-mail conforme a LGPD.
              </span>
            </label>
            <motion.button
              whileHover={{ scale: checked ? 1.02 : 1 }}
              whileTap={{ scale: checked ? 0.98 : 1 }}
              className="mt-8 w-full rounded-full bg-brand px-5 py-3 text-sm font-semibold text-white shadow-brand/40 transition disabled:cursor-not-allowed disabled:bg-slate-300"
              onClick={handleConfirm}
              disabled={!checked}
            >
              Confirmar consentimento
            </motion.button>
            <p className="mt-4 text-xs text-slate-400">
              Guardamos apenas as informações essenciais para o relacionamento com a clínica.
            </p>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
