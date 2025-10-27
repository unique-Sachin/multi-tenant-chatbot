'use client';

import { useState, useEffect } from 'react';
import { useAuth } from '@/components/auth/auth-provider';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { 
  MessageSquare, 
  Plus, 
  MoreVertical, 
  Trash2, 
  LogOut, 
  User, 
  Building2,
  Upload,
  Settings
} from 'lucide-react';
import { cn } from '@/lib/utils';
import Link from 'next/link';
import type { Organization, Website, Conversation } from '@/lib/types';
import { chatService } from '@/lib/services/chat';

interface ChatSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  namespace: string;
  onNamespaceChange: (namespace: string) => void;
  onNewConversation: () => void;
  onConversationClick: (conversation: Conversation) => void;
  currentSessionId?: string;
  refreshTrigger?: number;
  organizations: Organization[];
  websites: Website[];
}

export function ChatSidebar({
  isOpen,
  onClose,
  namespace,
  onNamespaceChange,
  onNewConversation,
  onConversationClick,
  currentSessionId,
  refreshTrigger,
  organizations,
  websites,
}: ChatSidebarProps) {
  const { user, logout } = useAuth();
  const [conversations, setConversations] = useState<Record<string, Conversation[]>>({});

  useEffect(() => {
    loadConversations();
  }, [refreshTrigger]);

  const loadConversations = async () => {
    try {
      const response = await chatService.getConversations();
      
      // Group by namespace
      const grouped: Record<string, Conversation[]> = {};
      response.conversations.forEach((conv) => {
        if (!grouped[conv.namespace]) {
          grouped[conv.namespace] = [];
        }
        grouped[conv.namespace].push(conv);
      });
      
      setConversations(grouped);
    } catch (error) {
      console.error('Failed to load conversations:', error);
    }
  };

  const handleDeleteConversation = async (conv: Conversation) => {
    try {
      await chatService.deleteConversation(conv.namespace, conv.session_id);
      await loadConversations();
    } catch (error) {
      console.error('Failed to delete conversation:', error);
    }
  };

  // Create namespace options
  const namespaceOptions = [
    { value: 'zibtek', label: 'Zibtek (default)' },
    ...websites
      .filter((website) => website.namespace !== 'zibtek') // Exclude duplicate zibtek
      .map((website) => {
        const org = organizations.find((o) => o.id === website.org_id);
        return {
          value: website.namespace,
          label: `${org?.name || 'Unknown'} - ${website.url}`,
        };
      }),
  ];

  const currentNamespaceConversations = conversations[namespace] || [];

  return (
    <>
      {/* Mobile Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-background/80 backdrop-blur-sm z-40 lg:hidden"
          onClick={onClose}
        />
      )}

      {/* Sidebar */}
      <div
        className={cn(
          'fixed lg:relative inset-y-0 left-0 z-50 w-80 bg-background border-r flex flex-col transition-transform duration-200',
          isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
        )}
      >
        {/* User Profile */}
        <div className="p-4 border-b">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" className="w-full justify-start gap-3 h-auto py-3">
                <Avatar className="h-10 w-10">
                  <AvatarFallback className="bg-primary text-primary-foreground">
                    {user?.email?.charAt(0).toUpperCase() || 'U'}
                  </AvatarFallback>
                </Avatar>
                <div className="flex-1 text-left overflow-hidden">
                  <p className="font-medium text-sm truncate">{user?.full_name || user?.email}</p>
                  <p className="text-xs text-muted-foreground truncate">{user?.email}</p>
                </div>
                <MoreVertical className="h-4 w-4 flex-shrink-0" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" className="w-56">
              <DropdownMenuItem asChild>
                <Link href="/profile" className="cursor-pointer">
                  <User className="mr-2 h-4 w-4" />
                  Profile
                </Link>
              </DropdownMenuItem>
              <DropdownMenuItem asChild>
                <Link href="/settings" className="cursor-pointer">
                  <Settings className="mr-2 h-4 w-4" />
                  Settings
                </Link>
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={logout} className="text-destructive cursor-pointer">
                <LogOut className="mr-2 h-4 w-4" />
                Logout
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>

        {/* Namespace Selector */}
        <div className="p-4 space-y-3">
          <div className="space-y-2">
            <label className="text-sm font-medium">Knowledge Base</label>
            <Select value={namespace} onValueChange={onNamespaceChange}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {namespaceOptions.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <Button onClick={onNewConversation} className="w-full" variant="default">
            <Plus className="mr-2 h-4 w-4" />
            New Conversation
          </Button>
        </div>

        <Separator />

        {/* Quick Links */}
        <div className="p-4 space-y-2">
          <Link href="/organizations">
            <Button variant="ghost" className="w-full justify-start" size="sm">
              <Building2 className="mr-2 h-4 w-4" />
              Organizations
            </Button>
          </Link>
          <Link href="/documents">
            <Button variant="ghost" className="w-full justify-start" size="sm">
              <Upload className="mr-2 h-4 w-4" />
              Documents
            </Button>
          </Link>
        </div>

        <Separator />

        {/* Conversations List */}
        <div className="flex-1 overflow-hidden">
          <div className="p-4 pb-2">
            <h3 className="text-sm font-medium text-muted-foreground">Recent Conversations</h3>
          </div>
          
          <ScrollArea className="h-full px-4">
            <div className="space-y-2 pb-4">
              {currentNamespaceConversations.length === 0 ? (
                <p className="text-sm text-muted-foreground text-center py-8">
                  No conversations yet
                </p>
              ) : (
                currentNamespaceConversations.slice(0, 20).map((conv) => {
                  const isActive = currentSessionId === conv.session_id;
                  return (
                    <div key={conv.id} className="group relative">
                      <Button
                        variant={isActive ? "secondary" : "ghost"}
                        className={cn(
                          "w-full justify-start text-left h-auto py-2 pr-10",
                          isActive && "bg-secondary"
                        )}
                        size="sm"
                        onClick={() => onConversationClick(conv)}
                      >
                        <MessageSquare className="mr-2 h-4 w-4 flex-shrink-0" />
                        <span className="truncate flex-1">
                          {conv.title || 'Untitled'}
                        </span>
                        </Button>
                      
                      <Button
                        variant="ghost"
                        size="icon"
                        className="absolute right-1 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity h-7 w-7"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteConversation(conv);
                        }}
                      >
                        <Trash2 className="h-3.5 w-3.5 text-destructive" />
                      </Button>
                    </div>
                  );
                })
              )}
            </div>
          </ScrollArea>
        </div>
      </div>
    </>
  );
}
