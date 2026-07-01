import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext.jsx';
import Layout from './components/Layout.jsx';
import LoginPage from './pages/LoginPage.jsx';
import DatasetsPage from './pages/DatasetsPage.jsx';
import DatasetDetailPage from './pages/DatasetDetailPage.jsx';
import ConnectorsPage from './pages/ConnectorsPage.jsx';
import ExperimentDetailPage from './pages/ExperimentDetailPage.jsx';
import RAGPage from './pages/RAGPage.jsx';
import EvaluationPage from './pages/EvaluationPage.jsx';
import FineTuningPage from './pages/FineTuningPage.jsx';
import ForecastingPage from './pages/ForecastingPage.jsx';
import AgentPage from './pages/AgentPage.jsx';
import AgentTrainingPage from './pages/AgentTrainingPage.jsx';
import MLOpsPage from './pages/MLOpsPage.jsx';
import ABTestingPage from './pages/ABTestingPage.jsx';
import SettingsPage from './pages/SettingsPage.jsx';

// Redirects to /login when the user is not authenticated.
// Shows a blank screen while the auth context is checking localStorage.
function ProtectedRoute({ children }) {
  const { isAuthenticated, loading } = useAuth();
  if (loading) return null;
  return isAuthenticated ? children : <Navigate to="/login" replace />;
}

function AppRoutes() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route element={
        <ProtectedRoute>
          <Layout />
        </ProtectedRoute>
      }>
        <Route index element={<Navigate to="/datasets" replace />} />
        <Route path="/datasets" element={<DatasetsPage />} />
        <Route path="/datasets/:id" element={<DatasetDetailPage />} />
        <Route path="/connectors" element={<ConnectorsPage />} />
        <Route path="/experiments" element={<Navigate to="/datasets" replace />} />
        <Route path="/experiments/:id" element={<ExperimentDetailPage />} />
        <Route path="/rag" element={<RAGPage />} />
        <Route path="/fine-tuning" element={<FineTuningPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/ab-testing" element={<ABTestingPage />} />
          <Route path="/agent" element={<AgentPage />} />
          <Route path="/agent-training" element={<AgentTrainingPage />} />
          <Route path="/mlops" element={<MLOpsPage />} />
          <Route path="/forecasting" element={<ForecastingPage />} />
          <Route path="/evaluation" element={<EvaluationPage />} />
      </Route>
    </Routes>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AppRoutes />
      </AuthProvider>
    </BrowserRouter>
  );
}
