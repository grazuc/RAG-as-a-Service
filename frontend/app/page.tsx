"use client"

import type React from "react"

import { useState } from "react"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Loader2, Search, BookOpen, FileText, AlertCircle } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"

interface Source {
  file: string
  snippet: string
}

interface ApiResponse {
  original_query: string
  rewritten_query: string
  answer: string
  sources: Source[]
}

export default function Home() {
  const [isLoading, setIsLoading] = useState(false)
  const [response, setResponse] = useState<ApiResponse | null>(null)
  const [query, setQuery] = useState("")
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim() || isLoading) return

    setIsLoading(true)
    setError(null)

    try {
      const res = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query }),
      })

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}))
        throw new Error(errorData.detail || `Error ${res.status}: ${res.statusText}`)
      }

      const data = await res.json()
      setResponse(data)
    } catch (error) {
      console.error("Error:", error)
      setError(error instanceof Error ? error.message : "Error al conectar con el servidor")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-b from-blue-50 to-white p-4 md:p-8">
      <div className="max-w-4xl mx-auto">
        <div className="flex flex-col items-center justify-center mb-8">
          <div className="bg-blue-100 p-3 rounded-full mb-4">
            <BookOpen className="h-8 w-8 text-blue-600" />
          </div>
          <h1 className="text-3xl font-bold text-center mb-2 text-blue-900">Asistente de Consulta Inteligente</h1>
          <p className="text-gray-600 text-center max-w-md">
            Realiza preguntas sobre cualquier tema y obtén respuestas precisas con fuentes verificadas
          </p>
        </div>

        <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
          <div className="p-6">
            <form onSubmit={handleSubmit} className="w-full">
              <div className="relative">
                <div className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400">
                  <Search className="h-5 w-5" />
                </div>
                <Input
                  type="text"
                  placeholder="¿Qué deseas saber? Escribe tu pregunta aquí..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  className="pl-10 pr-24 py-6 text-base rounded-full border-gray-200 shadow-sm"
                  disabled={isLoading}
                />
                <div className="absolute right-1.5 top-1/2 -translate-y-1/2">
                  <Button
                    type="submit"
                    disabled={isLoading || !query.trim()}
                    className="rounded-full px-4 bg-blue-600 hover:bg-blue-700 text-white"
                  >
                    {isLoading ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <>
                        <span className="mr-2">Enviar</span>
                        <svg
                          className="h-4 w-4"
                          xmlns="http://www.w3.org/2000/svg"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        >
                          <line x1="22" y1="2" x2="11" y2="13"></line>
                          <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                        </svg>
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </form>
          </div>

          {error && (
            <div className="px-6 pb-6">
              <Alert variant="destructive" className="bg-red-50 text-red-800 border-red-200">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            </div>
          )}

          {(isLoading || response) && (
            <div className="border-t border-gray-200">
              <div className="p-6">
                <div className="flex items-center gap-2 mb-3">
                  <div className="bg-blue-100 p-1.5 rounded-full">
                    <svg
                      className="h-5 w-5 text-blue-600"
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <circle cx="12" cy="12" r="10"></circle>
                      <line x1="12" y1="16" x2="12" y2="12"></line>
                      <line x1="12" y1="8" x2="12.01" y2="8"></line>
                    </svg>
                  </div>
                  <h2 className="font-medium text-lg">Respuesta</h2>
                </div>

                <Card className="border-0 bg-gray-50 overflow-hidden">
                  <div className="p-4">
                    {isLoading ? (
                      <div className="flex flex-col items-center justify-center py-8">
                        <div className="relative">
                          <div className="h-16 w-16 rounded-full border-4 border-gray-200 border-t-blue-600 animate-spin"></div>
                          <div className="absolute inset-0 flex items-center justify-center">
                            <Loader2 className="h-8 w-8 text-blue-600 animate-pulse" />
                          </div>
                        </div>
                        <p className="mt-4 text-gray-500 text-center">
                          Buscando respuesta para: <span className="font-medium text-gray-700">"{query}"</span>
                        </p>
                      </div>
                    ) : (
                      <div className="animate-fade-in">
                        <p className="whitespace-pre-line text-base leading-relaxed text-gray-700">
                          {response?.answer}
                        </p>
                        {response?.rewritten_query !== response?.original_query && (
                          <div className="mt-4 text-xs text-gray-500 border-t border-gray-200 pt-2">
                            <span className="font-medium">Consulta optimizada:</span> {response?.rewritten_query}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </Card>
              </div>

              {response && response.sources.length > 0 && (
                <div className="border-t border-gray-200">
                  <div className="p-6">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="bg-amber-100 p-1.5 rounded-full">
                        <svg
                          className="h-5 w-5 text-amber-600"
                          xmlns="http://www.w3.org/2000/svg"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        >
                          <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path>
                          <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path>
                        </svg>
                      </div>
                      <h2 className="font-medium text-lg">Fuentes consultadas</h2>
                    </div>

                    <div className="grid gap-4 md:grid-cols-2">
                      {response.sources.map((source, index) => (
                        <Card
                          key={index}
                          className="h-full overflow-hidden hover:shadow-md transition-shadow duration-300 border-gray-200"
                        >
                          <div className="bg-gray-50 p-3 border-b border-gray-200 flex items-center gap-2">
                            <FileText className="h-4 w-4 text-gray-500" />
                            <p className="font-medium text-sm truncate">{source.file}</p>
                          </div>
                          <div className="p-4">
                            <p className="text-sm text-gray-600 line-clamp-4 leading-relaxed">{source.snippet}</p>
                          </div>
                        </Card>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </main>
  )
}
