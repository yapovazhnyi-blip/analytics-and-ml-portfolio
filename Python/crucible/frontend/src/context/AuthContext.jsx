import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import api from '../api/client.js';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);        // { id, email, role }
  const [loading, setLoading] = useState(true);  // checking stored token on mount

  // ── Token storage helpers ───────────────────────────────────────────────
  const getToken  = () => localStorage.getItem('crucible_access_token');
  const saveTokens = (access, refresh) => {
    localStorage.setItem('crucible_access_token', access);
    localStorage.setItem('crucible_refresh_token', refresh);
  };
  const clearTokens = () => {
    localStorage.removeItem('crucible_access_token');
    localStorage.removeItem('crucible_refresh_token');
  };

  // ── Axios interceptors ──────────────────────────────────────────────────
  useEffect(() => {
    // Attach access token to every request
    const reqId = api.interceptors.request.use(config => {
      const token = getToken();
      if (token) config.headers.Authorization = `Bearer ${token}`;
      return config;
    });

    // On 401, try to refresh; on failure, log out
    const resId = api.interceptors.response.use(
      res => res,
      async err => {
        if (err.response?.status === 401 && !err.config._retry) {
          err.config._retry = true;
          const refresh = localStorage.getItem('crucible_refresh_token');
          if (refresh) {
            try {
              const { data } = await api.post('/auth/refresh', { refresh_token: refresh });
              const { access_token, refresh_token } = data.data;
              saveTokens(access_token, refresh_token);
              err.config.headers.Authorization = `Bearer ${access_token}`;
              return api(err.config);
            } catch {
              logout();
            }
          } else {
            logout();
          }
        }
        return Promise.reject(err);
      }
    );

    return () => {
      api.interceptors.request.eject(reqId);
      api.interceptors.response.eject(resId);
    };
  }, []);

  // ── Restore session from localStorage on page load ──────────────────────
  useEffect(() => {
    const token = getToken();
    if (token) {
      api.get('/auth/me')
        .then(res => setUser(res.data.data))
        .catch(() => { clearTokens(); setUser(null); })
        .finally(() => setLoading(false));
    } else {
      setLoading(false);
    }
  }, []);

  // ── Auth actions ────────────────────────────────────────────────────────
  const login = useCallback(async (email, password) => {
    const { data } = await api.post('/auth/login', { email, password });
    const { access_token, refresh_token, user_id, email: userEmail, role } = data.data;
    saveTokens(access_token, refresh_token);
    setUser({ id: user_id, email: userEmail, role });
    return data.data;
  }, []);

  const register = useCallback(async (email, password) => {
    const { data } = await api.post('/auth/register', { email, password });
    const { access_token, refresh_token, user_id, email: userEmail, role } = data.data;
    saveTokens(access_token, refresh_token);
    setUser({ id: user_id, email: userEmail, role });
    return data.data;
  }, []);

  const logout = useCallback(() => {
    clearTokens();
    setUser(null);
  }, []);

  const isAuthenticated = Boolean(user);
  const isAdmin = user?.role === 'admin';

  return (
    <AuthContext.Provider value={{ user, loading, isAuthenticated, isAdmin, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>');
  return ctx;
}
