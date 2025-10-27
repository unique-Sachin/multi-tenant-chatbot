'use client';

import { Message as MessageType } from '@/lib/types';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { User, Bot, ExternalLink, Clock, CheckCircle2, AlertCircle } from 'lucide-react';

interface ChatMessageProps {
  message: MessageType;
  isStreaming?: boolean;
}

export function ChatMessage({ message, isStreaming = false }: ChatMessageProps) {
  const isUser = message.is_user;

  return (
    <div className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'} mb-4 animate-in fade-in slide-in-from-bottom-2 duration-300`}>
      {!isUser && (
        <div className="flex-shrink-0">
          <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
            <Bot className="w-5 h-5 text-primary" />
          </div>
        </div>
      )}

      <div className={`flex-1 max-w-[80%] ${isUser ? 'flex justify-end' : ''}`}>
        <Card className={`p-4 ${isUser ? 'bg-primary text-primary-foreground' : 'bg-muted'}`}>
          <div className="prose prose-sm dark:prose-invert max-w-none">
            {message.content}
            {isStreaming && (
              <span className="inline-block w-2 h-4 ml-1 bg-current animate-pulse" />
            )}
          </div>

          {!isUser && !isStreaming && (
            <>
              {/* Citations */}
              {message.citations && message.citations.length > 0 && (
                <div className="mt-4 pt-4 border-t border-border">
                  <p className="text-xs font-semibold mb-2 flex items-center gap-1">
                    <ExternalLink className="w-3 h-3" />
                    Sources
                  </p>
                  <div className="space-y-1">
                    {message.citations.map((citation, idx) => (
                      <a
                        key={idx}
                        href={citation}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs text-blue-600 dark:text-blue-400 hover:underline block truncate"
                      >
                        {idx + 1}. {citation}
                      </a>
                    ))}
                  </div>
                </div>
              )}

              {/* Metadata */}
              <div className="mt-3 flex flex-wrap gap-2 text-xs text-muted-foreground">
                {message.processing_time_ms && (
                  <Badge variant="secondary" className="gap-1">
                    <Clock className="w-3 h-3" />
                    {message.processing_time_ms}ms
                  </Badge>
                )}
                
                {message.is_out_of_scope !== undefined && (
                  <Badge variant={message.is_out_of_scope ? "destructive" : "default"} className="gap-1">
                    {message.is_out_of_scope ? (
                      <>
                        <AlertCircle className="w-3 h-3" />
                        Refusal
                      </>
                    ) : (
                      <>
                        <CheckCircle2 className="w-3 h-3" />
                        Grounded
                      </>
                    )}
                  </Badge>
                )}

                {message.intent && (
                  <Badge variant="outline" className="gap-1">
                    {message.intent.replace('_', ' ')}
                    {message.intent_confidence && ` (${(message.intent_confidence * 100).toFixed(0)}%)`}
                  </Badge>
                )}
              </div>
            </>
          )}
        </Card>
      </div>

      {isUser && (
        <div className="flex-shrink-0">
          <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
            <User className="w-5 h-5 text-primary-foreground" />
          </div>
        </div>
      )}
    </div>
  );
}
