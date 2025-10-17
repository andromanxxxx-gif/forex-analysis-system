import React, { useState } from 'react';
import { useAnalysis, useMarketStatus } from '../hooks/useAnalysis';
import PriceDisplay from '../components/common/PriceDisplay';
import PriceChart from '../components/charts/PriceChart';
import TechnicalAnalysis from '../components/analysis/TechnicalAnalysis';
import AIAnalysis from '../components/analysis/AIAnalysis';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorDisplay from '../components/common/ErrorDisplay';
import { RefreshCw, Clock } from 'lucide-react';

const Dashboard: React.FC = () => {
  const [timeframe, setTimeframe] = useState('1D');
  const { data: analysis, isLoading, error, refetch } = useAnalysis(timeframe);
  const { data: marketStatus } = useMarketStatus();

  if (error) {
    return (
      <ErrorDisplay
        title="Failed to load analysis"
        message={error.message}
        onRetry={() => refetch()}
      />
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">XAUUSD Analysis</h1>
          <p className="text-gray-600">Real-time gold price analysis with AI insights</p>
        </div>
        
        <div className="flex items-center space-x-4 mt-4 lg:mt-0">
          <select
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value)}
            className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-orange-500"
          >
            <option value="1D">Daily</option>
            <option value="4H">4 Hours</option>
            <option value="1H">1 Hour</option>
          </select>

          <button
            onClick={() => refetch()}
            disabled={isLoading}
            className="flex items-center space-x-2 px-4 py-2 bg-orange-500 text-white rounded-lg hover:bg-orange-600 disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {isLoading && !analysis && (
        <div className="flex items-center justify-center h-64">
          <LoadingSpinner size="lg" text="Loading market analysis..." />
        </div>
      )}

      {analysis && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-900">Current Price</h2>
                <div className="flex items-center text-sm text-gray-500">
                  <Clock className="w-4 h-4 mr-1" />
                  {new Date(analysis.timestamp).toLocaleTimeString()}
                </div>
              </div>
              <PriceDisplay currentPrice={analysis.current_price} size="lg" />
              <div className="mt-4 text-sm text-gray-600">
                Symbol: {analysis.symbol} | Timeframe: {timeframe}
              </div>
            </div>

            {marketStatus && (
              <>
                <div className="bg-white rounded-lg border border-gray-200 p-6">
                  <h3 className="text-sm font-semibold text-gray-700 mb-4">Daily Overview</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Trend</span>
                      <span className={`font-semibold ${
                        marketStatus['1D'].trend === 'BULLISH' 
                          ? 'text-green-600' 
                          : marketStatus['1D'].trend === 'BEARISH'
                          ? 'text-red-600'
                          : 'text-gray-600'
                      }`}>
                        {marketStatus['1D'].trend}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Confidence</span>
                      <span className="font-semibold text-gray-900">
                        {(marketStatus['1D'].confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>

                <div className="bg-white rounded-lg border border-gray-200 p-6">
                  <h3 className="text-sm font-semibold text-gray-700 mb-4">Key Levels</h3>
                  <div className="space-y-2">
                    <div>
                      <div className="text-xs text-green-600 font-medium">Nearest Support</div>
                      <div className="text-lg font-semibold text-gray-900">
                        ${marketStatus['1D'].support_levels[0]?.toFixed(2) || 'N/A'}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-red-600 font-medium">Nearest Resistance</div>
                      <div className="text-lg font-semibold text-gray-900">
                        ${marketStatus['1D'].resistance_levels[0]?.toFixed(2) || 'N/A'}
                      </div>
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>

          <PriceChart
            data={analysis.chart_data?.plotly_config?.data[0]?.x?.map((timestamp: string, index: number) => ({
              timestamp,
              close: analysis.chart_data?.plotly_config?.data[0]?.y[index]
            })) || []}
            supportLevels={analysis.technical_analysis.support_levels}
            resistanceLevels={analysis.technical_analysis.resistance_levels}
          />

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            <TechnicalAnalysis analysis={analysis.technical_analysis} />
            <AIAnalysis analysis={analysis.ai_analysis} />
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
