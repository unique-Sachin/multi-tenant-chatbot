// Authentication context and provider for managing user state

'use client';

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { authService } from '@/lib/services/auth';
import type { User, LoginCredentials, SignupCredentials } from '@/lib/types';

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (credentials: LoginCredentials) => Promise<void>;
  signup: (credentials: SignupCredentials) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();

  // Check for existing session on mount
  useEffect(() => {
    const token = authService.getToken();
    if (token) {
      // Token exists, verify it's valid by fetching user data
      // For now, we'll just set isLoading to false
      // In production, you'd want to validate the token
      setIsLoading(false);
    } else {
      setIsLoading(false);
    }
  }, []);

  const login = useCallback(async (credentials: LoginCredentials) => {
    const response = await authService.login(credentials);
    setUser(response.user);
    router.push('/chat');
  }, [router]);

  const signup = useCallback(async (credentials: SignupCredentials) => {
    const response = await authService.signup(credentials);
    setUser(response.user);
    router.push('/chat');
  }, [router]);

  const logout = useCallback(() => {
    authService.logout();
    setUser(null);
    router.push('/login');
  }, [router]);

  const value = {
    user,
    isLoading,
    isAuthenticated: !!user || authService.isAuthenticated(),
    login,
    signup,
    logout,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
