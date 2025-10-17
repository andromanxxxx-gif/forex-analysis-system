import React from 'react';
import { TechnicalAnalysis as TechnicalAnalysisType } from '../../types/analysis';
import { getSignalColor, formatNumber } from '../../utils/formatting';
import { TrendingUp, TrendingDown, Minus, Target } from 'lucide-react';

interface TechnicalAnalysisProps {
  analysis: TechnicalAnalysisType;
}

const TechnicalAnalysis: React.FC<TechnicalAnalysisProps> = ({ analysis }) => {
  const getTrendIcon = (signal: string) => {
    switch (signal) {
      case 'BUY':
        return <TrendingUp className="w-4 h-4" />;
      case 'SELL':
        return <TrendingDown className="w-4 h-4" />;
      default:
        return <Minus className="w-4 h-4" />;
    }
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
        <Target className="w-5 h-5 mr-2 text-orange-500" />
        Technical Analysis
      </h2>

      <div className="mb-6">
        <div className={`inline-flex items-center px-4 py-2 rounded-full text-sm font-medium ${
          analysis.summary === 'BULLISH' 
            ? 'bg-green-100 text-green-800'
            : analysis.summary === 'BEARISH'
            ? 'bg-red-100 text-red-800'
            : 'bg-gray-100 text-gray-800'
        }`}>
          {getTrendIcon(analysis.summary === 'BULLISH' ? 'BUY' : analysis.summary === 'BEARISH' ? 'SELL' : 'NEUTRAL')}
          <span className="ml-2">
            {analysis.summary} - {(analysis.confidence * 100).toFixed(1)}% Confidence
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
        {analysis.indicators.map((indicator, index) => (
          <div
            key={index}
            className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700">
                {indicator.name}
              </span>
              <div className={`px-2 py-1 rounded text-xs font-medium ${getSignalColor(indicator.signal)}`}>
                {indicator.signal}
              </div>
            </div>
            <div className="text-lg font-semibold text-gray-900">
              {formatNumber(indicator.value)}
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Support Levels</h3>
          <div className="space-y-2">
            {analysis.support_levels.map((level, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-3 bg-green-50 border border-green-200 rounded-lg"
              >
                <span className="text-green-700 font-medium">S{index + 1}</span>
                <span className="text-green-900 font-semibold">
                  ${level.toFixed(2)}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Resistance Levels</h3>
          <div className="space-y-2">
            {analysis.resistance_levels.map((level, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-3 bg-red-50 border border-red-200 rounded-lg"
              >
                <span className="text-red-700 font-medium">R{index + 1}</span>
                <span className="text-red-900 font-semibold">
                  ${level.toFixed(2)}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TechnicalAnalysis;
