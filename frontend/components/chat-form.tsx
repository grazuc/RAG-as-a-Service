"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Loader2, Search, SendIcon } from "lucide-react"

interface ChatFormProps {
  onSubmit: (query: string) => Promise<void>
  isLoading: boolean
}

export function ChatForm({ onSubmit, isLoading }: ChatFormProps) {
  const [query, setQuery] = useState("")

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim() && !isLoading) {
      onSubmit(query)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="w-full">
      <div className="relative">
        <div className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400">
          <Search className="h-5 w-5" />
        </div>
        <Input
          type="text"
          placeholder="¿Qué deseas saber? Escribe tu pregunta aquí..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="pl-10 pr-24 py-6 text-base rounded-full border-slate-200 dark:border-slate-700 shadow-sm focus-visible:ring-primary"
          disabled={isLoading}
        />
        <div className="absolute right-1.5 top-1/2 -translate-y-1/2">
          <Button type="submit" disabled={isLoading || !query.trim()} size="sm" className="rounded-full px-4">
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <>
                <span className="mr-2">Enviar</span>
                <SendIcon className="h-3.5 w-3.5" />
              </>
            )}
          </Button>
        </div>
      </div>
    </form>
  )
}
