"use client"

import { Card, CardContent } from "@/components/ui/card"
import { FileText, LinkIcon } from "lucide-react"
import { motion } from "framer-motion"

interface Source {
  file: string
  snippet: string
}

interface SourcesListProps {
  sources: Source[]
}

export function SourcesList({ sources }: SourcesListProps) {
  return (
    <div className="p-6">
      <div className="flex items-center gap-2 mb-4">
        <div className="bg-amber-500/10 p-1.5 rounded-full">
          <LinkIcon className="h-5 w-5 text-amber-500" />
        </div>
        <h2 className="font-medium text-lg">Fuentes consultadas</h2>
      </div>

      <motion.div
        className="grid gap-4 md:grid-cols-2"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ staggerChildren: 0.1 }}
      >
        {sources.map((source, index) => (
          <SourceCard key={index} source={source} index={index} />
        ))}
      </motion.div>
    </div>
  )
}

interface SourceCardProps {
  source: Source
  index: number
}

function SourceCard({ source, index }: SourceCardProps) {
  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.1 }}>
      <Card className="h-full overflow-hidden hover:shadow-md transition-shadow duration-300 border-slate-200 dark:border-slate-700">
        <CardContent className="p-0">
          <div className="bg-slate-50 dark:bg-slate-800 p-3 border-b border-slate-200 dark:border-slate-700 flex items-center gap-2">
            <FileText className="h-4 w-4 text-slate-500" />
            <p className="font-medium text-sm truncate">{source.file}</p>
          </div>
          <div className="p-4">
            <p className="text-sm text-slate-600 dark:text-slate-300 line-clamp-4 leading-relaxed">{source.snippet}</p>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
