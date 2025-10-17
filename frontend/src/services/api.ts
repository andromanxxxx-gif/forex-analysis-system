import axios from 'axios';
import { AnalysisResponse, MarketStatus } from '../types/analysis';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

export const analysisAPI = {
  getXAUUSDAnalysis: async (timeframe: string = '1D'): Promise<AnalysisResponse> => {
    const response = await api.get<AnalysisResponse>(`/analysis/xauusd?timeframe=${timeframe}`);
    return response.data;
  },

  getHistoricalAnalysis: async (timeframe: string = '1D', period: string = '2y') => {
    const response = await api.get(`/analysis/historical?timeframe=${timeframe}&period=${period}`);
    return response.data;
  },

  getMarketStatus: async (): Promise<MarketStatus> => {
    const response = await api.get<MarketStatus>('/market/status');
    return response.data;
  },
};
