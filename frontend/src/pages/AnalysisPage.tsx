import React, { useState } from 'react';
import { useHistoricalAnalysis } from '../hooks/useAnalysis';
import PriceChart from '../components/charts/PriceChart';
import { Download, Calendar } from 'lucide-react';

const AnalysisPage: React.FC = () => {
  const [timeframe, setTimeframe] = useState('1D');
  const [period, setPeriod] = useState('2y');
  
  const { data: historicalData, isLoading } = useHistoricalAnalysis(timeframe, period);

  const periodOptions = {
    '1D': [
      { value: '6m', label: '6 Months' },
      { value: '1y', label: '1 Year' },
      { value: '2y', label: '2 Years' }
    ],
    '4H': [
      { value: '6m', label: '6 Months' },
      { value: '1y', label: '1 Year' }
    ],
    '1H': [
      { value: '6m', label: '6 Months' }
    ]
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Historical Analysis</h1>
          <p className="text-gray-600">Deep dive into historical price data and patterns</p>
        </div>
        
        <div className="flex items-center space-x-4 mt-4 lg:mt-0">
          <select
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value)}
            className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-orange-500"
          >
            <option value="1D">Daily Chart</option>
            <option value="4H">4 Hour Chart</option>
            <option value="1H">1 Hour Chart</option>
          </select>

          <select
            value={period}
            onChange={(e) => setPeriod(e.target.value)}
            className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-orange-500"
          >
            {periodOptions[timeframe as keyof typeof periodOptions]?.map(option => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>

          <button className="flex items-center space-x-2 px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50">
            <Download className="w-4 h-4" />
            <span>Export</span>
          </button>
        </div>
      </div>

      {isLoading && (
        <div className="flex items-center justify-center h-96">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-orange-500 mx-auto mb-4"></div>
            <p className="text-gray-600">Loading historical data...</p>
          </div>
        </div>
      )}

      {historicalData && (
        <div className="space-y-6">
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-900 flex items-center">
                <Calendar className
