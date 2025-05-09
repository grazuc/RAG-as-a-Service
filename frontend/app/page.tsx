"use client";

import React, { useState, useRef, useEffect } from "react";
import { Toaster, toast } from "sonner";
import ReactMarkdown from "react-markdown";

export default function Home() {
  const [documentUploaded, setDocumentUploaded] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<any>(null);
  const [conversations, setConversations] = useState<any[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [fileName, setFileName] = useState<string>("");

  // Cargar conversaciones desde localStorage al subir un archivo
  useEffect(() => {
    if (documentUploaded) {
      const saved = localStorage.getItem("conversations");
      if (saved) setConversations(JSON.parse(saved));
      const savedFile = localStorage.getItem("fileName");
      if (savedFile) setFileName(savedFile);
    }
  }, [documentUploaded]);

  // Guardar conversaciones en localStorage cuando cambian
  useEffect(() => {
    if (documentUploaded) {
      localStorage.setItem("conversations", JSON.stringify(conversations));
    }
  }, [conversations, documentUploaded]);

  // --- File Upload Logic ---
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!["pdf", "docx", "txt"].includes(file.name.split(".").pop()?.toLowerCase() || "")) {
      toast.error("Solo se permiten archivos PDF, DOCX o TXT");
      return;
    }
    setUploading(true);
    setResponse(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Error al subir el documento");
      }
      toast.success("Documento procesado correctamente");
      setDocumentUploaded(true);
      setConversations([]); // Limpiar conversaciones al subir nuevo archivo
      localStorage.removeItem("conversations");
      setFileName(file.name); // Guardar nombre del archivo
      localStorage.setItem("fileName", file.name);
    } catch (err: any) {
      toast.error(err.message || "Error al subir el documento");
    } finally {
      setUploading(false);
    }
  };

  // --- Query Logic ---
  const handleQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setResponse(null);
    try {
      const res = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query.trim() }),
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Error en la consulta");
      }
      const data = await res.json();
      setResponse(data);
      setConversations(prev => [
        ...prev,
        { question: query.trim(), answer: data.answer, sources: data.sources }
      ]);
    } catch (err: any) {
      toast.error(err.message || "Error en la consulta");
    } finally {
      setLoading(false);
    }
  };

  // Deduplicar fuentes por source y page
  const dedupedSources = (sources: any[] = []) => {
    const map = new Map();
    sources.forEach((src) => {
      const key = `${src.source}-${src.page ?? ""}`;
      if (!map.has(key)) {
        map.set(key, src);
      }
    });
    return Array.from(map.values());
  };

  function formatAnswer(text: string): string {
    return text.replace(/\n{2,}/g, '\n\n&nbsp;\n\n'); // truco para romper <p> y simular separación real
  };
  

  return (
    <main className="min-h-screen bg-background flex flex-row">
      {/* Sidebar de conversaciones solo si hay documento */}
      {documentUploaded && (
        <aside className="w-80 min-h-screen bg-muted/40 border-r border-muted-foreground/20 flex flex-col p-4">
          <h2 className="text-lg font-bold mb-4">Conversaciones</h2>
          <div className="flex-1 overflow-y-auto space-y-4">
            <div className="bg-background rounded-lg p-3 shadow">
              <div className="font-bold text-primary mb-2">{fileName}</div>
              {conversations.length > 0 ? (
                <div className="text-sm text-muted-foreground line-clamp-3">{conversations[conversations.length-1].question}</div>
              ) : (
                <div className="text-muted-foreground text-sm">No hay preguntas aún.</div>
              )}
            </div>
          </div>
        </aside>
      )}
      {/* Chat principal */}
      <div className="flex-1 flex flex-col items-center justify-center">
        <div className="w-full max-w-2xl mx-auto mt-12 p-6 rounded-2xl bg-muted/60 shadow-lg">
          <h1 className="text-3xl font-bold text-center mb-8">¿En qué puedo ayudarte?</h1>
          {/* Chat y respuesta */}
          <form onSubmit={handleQuery} className="flex flex-col gap-4">
            <textarea
              className="w-full rounded-lg border border-input bg-background px-4 py-3 text-base shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/60 min-h-[60px] resize-none"
              placeholder={documentUploaded ? "Escribe tu pregunta sobre el documento..." : "Sube un archivo para comenzar"}
              value={query}
              onChange={e => setQuery(e.target.value)}
              disabled={loading || !documentUploaded}
            />
            <div className="flex flex-row justify-between items-center gap-2">
              <button
                type="button"
                className="flex items-center gap-2 px-5 py-2 rounded-full bg-primary text-primary-foreground font-semibold shadow hover:bg-primary/90 transition disabled:opacity-60"
                onClick={() => fileInputRef.current?.click()}
                disabled={uploading}
              >
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 9V5.25A2.25 2.25 0 0013.5 3h-3A2.25 2.25 0 008.25 5.25V9m7.5 0v10.5A2.25 2.25 0 0113.5 21h-3A2.25 2.25 0 018.25 19.5V9m7.5 0H8.25" />
                </svg>
                {documentUploaded ? "Adjuntar nuevo archivo" : "Adjuntar archivo"}
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.docx,.txt"
                className="hidden"
                onChange={handleFileChange}
                disabled={uploading}
              />
              <button
                type="submit"
                className="px-6 py-2 rounded-full bg-primary text-primary-foreground font-semibold shadow hover:bg-primary/90 transition disabled:opacity-60"
                disabled={loading || !query.trim() || !documentUploaded}
              >
                {loading ? "Consultando..." : "Consultar"}
              </button>
            </div>
          </form>
          {response && (
            <div className="mt-8 space-y-6">
              <div className="bg-card rounded-xl p-5 shadow">
                <h2 className="font-semibold mb-2">Respuesta</h2>
                <div className="prose prose-neutral dark:prose-invert" style={{ margin: 0, padding: 0}}>
                  <style>{`
                    .prose p { margin: 0 !important; }
                    .prose h1, .prose h2, .prose h3, .prose h4, .prose h5, .prose h6 { margin: 0 !important; }
                    .prose h1 + p,
                    .prose h2 + p,
                    .prose h3 + p,
                    .prose h4 + p,
                    .prose h5 + p,
                    .prose h6 + p { margin-top: 0 !important; }
                  `}</style>
                  <ReactMarkdown skipHtml={false}>
                    {formatAnswer(response.answer)}
                  </ReactMarkdown>
                </div>
              </div>
              {response.sources && response.sources.length > 0 && (
                <div className="space-y-2">
                  <h3 className="font-semibold">Fuentes</h3>
                  {dedupedSources(response.sources).map((src: any, idx: number) => (
                    <div key={idx} className="bg-muted rounded-lg p-4">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-sm font-medium">{src.source}</span>
                        {src.page && (
                          <span className="text-xs text-muted-foreground">Página {src.page}</span>
                        )}
                      </div>
                      {src.content_preview && (
                        <p className="text-xs text-muted-foreground">{src.content_preview}</p>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
          {uploading && (
            <div className="text-center text-primary animate-pulse mt-4">Procesando documento...</div>
          )}
        </div>
        <Toaster />
      </div>
    </main>
  );
}
