"use client"

import { Card, CardContent } from "@/components/ui/card"
import { Loader2, Bot } from "lucide-react"
import { motion } from "framer-motion"

interface ResponseDisplayProps {
  answer?: string
  isLoading: boolean
  query: string
}

export function ResponseDisplay({ answer, isLoading, query }: ResponseDisplayProps) {
  return (
    <div className="p-6">
      <div className="flex items-center gap-2 mb-3">
        <div className="bg-primary/10 p-1.5 rounded-full">
          <Bot className="h-5 w-5 text-primary" />
        </div>
        <h2 className="font-medium text-lg">Respuesta</h2>
      </div>

      <Card className="border-0 bg-slate-50 dark:bg-slate-800/50 overflow-hidden">
        <CardContent className="p-4">
          {isLoading ? (
            <div className="flex flex-col items-center justify-center py-8">
              <div className="relative">
                <div className="h-16 w-16 rounded-full border-4 border-slate-200 dark:border-slate-700 border-t-primary animate-spin"></div>
                <div className="absolute inset-0 flex items-center justify-center">
                  <Loader2 className="h-8 w-8 text-primary animate-pulse" />
                </div>
              </div>
              <p className="mt-4 text-muted-foreground text-center">
                Buscando respuesta para: <span className="font-medium text-foreground">"{query}"</span>
              </p>
            </div>
          ) : (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="prose prose-slate dark:prose-invert max-w-none"
            >
              <p className="whitespace-pre-line text-base leading-relaxed">{answer}</p>
            </motion.div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
