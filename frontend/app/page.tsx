'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/components/auth/auth-provider';
import { Button } from "@/components/ui/button";
import { Bot, Sparkles, Shield, Zap } from 'lucide-react';
import Link from 'next/link';

export default function Home() {
  const { isAuthenticated, isLoading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isLoading && isAuthenticated) {
      router.push('/chat');
    }
  }, [isAuthenticated, isLoading, router]);

  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-slate-100 dark:from-slate-950 dark:via-blue-950 dark:to-slate-900">
      <div className="container mx-auto px-4 py-16">
        {/* Hero Section */}
        <div className="text-center space-y-6 mb-16">
          <div className="flex justify-center mb-6">
            <div className="p-4 bg-primary/10 rounded-full">
              <Bot className="h-16 w-16 text-primary" />
            </div>
          </div>
          
          <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            Multi-Tenant AI Assistant
          </h1>
          
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Enterprise-grade AI chatbot with isolated knowledge bases for multiple organizations
          </p>

          <div className="flex gap-4 justify-center pt-4">
            <Link href="/login">
              <Button size="lg" className="text-lg px-8">
                Get Started
              </Button>
            </Link>
            <Link href="/chat">
              <Button size="lg" variant="outline" className="text-lg px-8">
                Try Demo
              </Button>
            </Link>
          </div>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
          <div className="p-6 bg-white dark:bg-slate-800 rounded-lg shadow-lg border">
            <div className="p-3 bg-blue-100 dark:bg-blue-900 rounded-lg w-fit mb-4">
              <Shield className="h-6 w-6 text-blue-600 dark:text-blue-300" />
            </div>
            <h3 className="text-xl font-semibold mb-2">Secure & Isolated</h3>
            <p className="text-muted-foreground">
              Each organization gets its own isolated knowledge base with partition-based multi-tenancy
            </p>
          </div>

          <div className="p-6 bg-white dark:bg-slate-800 rounded-lg shadow-lg border">
            <div className="p-3 bg-purple-100 dark:bg-purple-900 rounded-lg w-fit mb-4">
              <Sparkles className="h-6 w-6 text-purple-600 dark:text-purple-300" />
            </div>
            <h3 className="text-xl font-semibold mb-2">Smart Retrieval</h3>
            <p className="text-muted-foreground">
              Hybrid search with vector similarity and keyword matching for accurate responses
            </p>
          </div>

          <div className="p-6 bg-white dark:bg-slate-800 rounded-lg shadow-lg border">
            <div className="p-3 bg-green-100 dark:bg-green-900 rounded-lg w-fit mb-4">
              <Zap className="h-6 w-6 text-green-600 dark:text-green-300" />
            </div>
            <h3 className="text-xl font-semibold mb-2">Real-time Streaming</h3>
            <p className="text-muted-foreground">
              See responses as they're generated with progress indicators and citations
            </p>
          </div>
        </div>

        {/* Tech Stack */}
        <div className="mt-16 text-center">
          <p className="text-sm text-muted-foreground mb-4">Powered by</p>
          <div className="flex flex-wrap justify-center gap-6 text-sm text-muted-foreground">
            <span>Next.js 15</span>
            <span>•</span>
            <span>OpenAI GPT-4</span>
            <span>•</span>
            <span>Milvus Vector DB</span>
            <span>•</span>
            <span>FastAPI</span>
            <span>•</span>
            <span>PostgreSQL</span>
          </div>
        </div>
      </div>
    </div>
  );
}
