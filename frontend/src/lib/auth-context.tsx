"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react";
import * as api from "./api";
import type { UserOut } from "./api";

type AuthState = {
  user: UserOut | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, name: string) => Promise<void>;
  logout: () => void;
};

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<UserOut | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const stored = api.getStoredUser();
    const token = api.getAccessToken();
    if (stored && token) {
      setUser(stored);
      api
        .me()
        .then((fresh) => setUser(fresh))
        .catch(() => {
          api.clearSession();
          setUser(null);
        })
        .finally(() => setLoading(false));
    } else {
      setLoading(false);
    }
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const tokens = await api.login(email, password);
    api.storeSession(tokens);
    setUser(tokens.user);
  }, []);

  const register = useCallback(
    async (email: string, password: string, name: string) => {
      const tokens = await api.register(email, password, name);
      api.storeSession(tokens);
      setUser(tokens.user);
    },
    []
  );

  const logout = useCallback(() => {
    api.clearSession();
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
