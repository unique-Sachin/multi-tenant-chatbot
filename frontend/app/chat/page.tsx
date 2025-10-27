'use client';

import { useState, useEffect, useRef } from 'react';
import { ProtectedRoute } from '@/components/auth/protected-route';
import { ChatMessage } from '@/components/chat/chat-message';
import { ChatInput } from '@/components/chat/chat-input';
import { ChatSidebar } from '@/components/chat/chat-sidebar';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Menu, Loader2 } from 'lucide-react';
import { chatService } from '@/lib/services/chat';
import { organizationService } from '@/lib/services/organizations';
import type { Message, StreamEvent, Website, Organization } from '@/lib/types';
import { toast } from 'sonner';

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentResponse, setCurrentResponse] = useState('');
  const [sessionId, setSessionId] = useState('');
  const [namespace, setNamespace] = useState('zibtek');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [organizations, setOrganizations] = useState<Organization[]>([]);
  const [websites, setWebsites] = useState<Website[]>([]);
  const [refreshSidebar, setRefreshSidebar] = useState(0);
  const [isLoadingConversation, setIsLoadingConversation] = useState(false);
  
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Generate new session ID on mount
  useEffect(() => {
    setSessionId(crypto.randomUUID());
  }, []);

  // Load organizations and websites
  useEffect(() => {
    loadOrganizations();
    loadWebsites();
  }, []);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, currentResponse]);

  const loadOrganizations = async () => {
    try {
      const orgs = await organizationService.getOrganizations();
      setOrganizations(orgs);
    } catch (error) {
      console.error('Failed to load organizations:', error);
    }
  };

  const loadWebsites = async () => {
    try {
      const sites = await organizationService.getWebsites();
      setWebsites(sites);
    } catch (error) {
      console.error('Failed to load websites:', error);
    }
  };

  const handleSendMessage = (content: string) => {
    // Add user message
    const userMessage: Message = {
      id: crypto.randomUUID(),
      content,
      is_user: true,
      timestamp: new Date().toISOString(),
    };
    
    setMessages(prev => [...prev, userMessage]);
    setIsStreaming(true);
    setCurrentResponse('');

    let finalResponse = '';
    let citations: string[] = [];
    let isOutOfScope = false;
    let processingTime = 0;
    let retrievalSteps: any = undefined;
    let intent: string | undefined = undefined;
    let intentConfidence: number | undefined = undefined;

    chatService.streamChat(
      {
        question: content,
        session_id: sessionId,
        partition_name: namespace,
      },
      (event: StreamEvent) => {
        switch (event.type) {
          case 'intent_detected':
            intent = event.intent;
            intentConfidence = event.confidence;
            toast.info(`Intent: ${event.intent} (${(event.confidence || 0) * 100}%)`, {
              duration: 2000,
            });
            break;

          case 'loading':
          case 'retrieval_progress':
          case 'reranking_progress':
            toast.info(event.message || 'Processing...', {
              duration: 1500,
            });
            break;

          case 'token':
            finalResponse = event.content || '';
            setCurrentResponse(finalResponse);
            break;

          case 'complete':
            citations = event.citations || [];
            isOutOfScope = event.is_out_of_scope || false;
            processingTime = event.processing_time_ms || 0;
            retrievalSteps = event.retrieval_steps;
            break;

          case 'error':
            toast.error(event.content || 'An error occurred');
            break;
        }
      },
      (error) => {
        console.error('Stream error:', error);
        toast.error('Failed to get response. Please try again.');
        setIsStreaming(false);
      },
      () => {
        // Complete
        const botMessage: Message = {
          id: crypto.randomUUID(),
          content: finalResponse || 'Sorry, I couldn\'t generate a response.',
          is_user: false,
          citations,
          is_out_of_scope: isOutOfScope,
          processing_time_ms: processingTime,
          retrieval_steps: retrievalSteps,
          timestamp: new Date().toISOString(),
          intent,
          intent_confidence: intentConfidence,
        };
        
        setMessages(prev => [...prev, botMessage]);
        setCurrentResponse('');
        setIsStreaming(false);
        
        // Refresh sidebar to show new conversation
        setRefreshSidebar(prev => prev + 1);
      }
    );
  };

  const handleNewConversation = () => {
    setSessionId(crypto.randomUUID());
    setMessages([]);
    setCurrentResponse('');
  };

  const handleConversationClick = async (conversation: any) => {
    try {
      setIsLoadingConversation(true);
      toast.info('Loading conversation...');
      
      // Load messages from backend
      const response = await chatService.getConversationMessages(
        conversation.namespace,
        conversation.session_id
      );
      
      // Set the session ID and namespace to match the conversation
      setSessionId(conversation.session_id);
      setNamespace(conversation.namespace);
      
      // Convert backend format to frontend Message type
      // Backend returns: { user_query, answer, citations, retrieval_steps, etc }
      // Frontend needs: alternating user/bot messages
      const loadedMessages: Message[] = [];
      
      response.messages.forEach((msg: any) => {
        // Add user message
        loadedMessages.push({
          id: `${msg.id}-user`,
          content: msg.user_query,
          is_user: true,
          timestamp: msg.created_at || new Date().toISOString(),
        });
        
        // Add bot response
        loadedMessages.push({
          id: `${msg.id}-bot`,
          content: msg.answer,
          is_user: false,
          timestamp: msg.created_at || new Date().toISOString(),
          citations: msg.citations || [],
          is_out_of_scope: msg.is_out_of_scope || false,
          processing_time_ms: msg.processing_time_ms || 0,
          retrieval_steps: msg.retrieval_steps,
        });
      });
      
      setMessages(loadedMessages);
      setCurrentResponse('');
      
      // Close sidebar on mobile
      setSidebarOpen(false);
      
      toast.success('Conversation loaded');
    } catch (error) {
      console.error('Failed to load conversation:', error);
      toast.error('Failed to load conversation history');
    } finally {
      setIsLoadingConversation(false);
    }
  };

  const handleNamespaceChange = (newNamespace: string) => {
    setNamespace(newNamespace);
    handleNewConversation();
  };

  return (
    <ProtectedRoute>
      <div className="flex h-screen bg-background">
        {/* Sidebar */}
        <ChatSidebar
          isOpen={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
          namespace={namespace}
          onNamespaceChange={handleNamespaceChange}
          onNewConversation={handleNewConversation}
          onConversationClick={handleConversationClick}
          currentSessionId={sessionId}
          refreshTrigger={refreshSidebar}
          organizations={organizations}
          websites={websites}
        />

        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Header */}
          <div className="border-b bg-background p-4 flex items-center gap-4 flex-shrink-0">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="lg:hidden"
            >
              <Menu className="h-5 w-5" />
            </Button>
            
            <div>
              <h1 className="text-xl font-semibold">Multi-Tenant AI Assistant</h1>
              <p className="text-sm text-muted-foreground">
                Knowledge Base: <span className="font-mono">{namespace}</span>
              </p>
            </div>
          </div>

          {/* Messages */}
          <ScrollArea className="flex-1 p-4 overflow-y-auto" ref={scrollAreaRef}>
            <div className="max-w-4xl mx-auto space-y-4">
              {isLoadingConversation ? (
                <div className="flex items-center justify-center py-8">
                  <div className="flex flex-col items-center gap-2">
                    <Loader2 className="h-8 w-8 animate-spin text-primary" />
                    <p className="text-sm text-muted-foreground">Loading conversation...</p>
                  </div>
                </div>
              ) : messages.length === 0 && !isStreaming ? (
                <Alert>
                  <AlertDescription>
                    👋 Hello! Ask me anything about the selected organization's content.
                  </AlertDescription>
                </Alert>
              ) : null}

              {!isLoadingConversation && messages.map((message) => (
                <ChatMessage key={message.id} message={message} />
              ))}

              {isStreaming && currentResponse && (
                <ChatMessage
                  message={{
                    id: 'streaming',
                    content: currentResponse,
                    is_user: false,
                    timestamp: new Date().toISOString(),
                  }}
                  isStreaming
                />
              )}

              <div ref={messagesEndRef} />
            </div>
          </ScrollArea>

          {/* Input */}
          <div className="flex-shrink-0">
            <ChatInput 
              onSend={handleSendMessage}
              disabled={false}
              isLoading={isStreaming}
            />
          </div>
        </div>
      </div>
    </ProtectedRoute>
  );
}
