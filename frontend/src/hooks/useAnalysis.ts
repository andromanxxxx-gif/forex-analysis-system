import { useQuery } from 'react-query';
import { analysisAPI } from '../services/api';
import { AnalysisResponse, MarketStatus } from '../types/analysis';

export const useAnalysis = (timeframe: string = '1D') => {
  return useQuery<AnalysisResponse, Error>(
    ['analysis', timeframe],
    () => analysisAPI.getXAUUSDAnalysis(timeframe),
    {
      refetchInterval: 60000,
      staleTime: 30000,
    }
  );
};

export const useMarketStatus = () => {
  return useQuery<MarketStatus, Error>(
    'market-status',
    analysisAPI.getMarketStatus,
    {
      refetchInterval: 30000,
    }
  );
};

export const useHistoricalAnalysis = (timeframe: string, period: string) => {
  return useQuery(
    ['historical', timeframe, period],
    () => analysisAPI.getHistoricalAnalysis(timeframe, period)
  );
};
