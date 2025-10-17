import React from 'react';
import { AIAnalysis as AIAnalysisType } from '../../types/analysis';
import { getRiskColor, getTrendColor } from '../../utils/formatting';
import { Brain, AlertTriangle, Target, TrendingUp } from 'lucide-react';

interface AIAnalysisProps {
  analysis: AIAnalysisType;
}

const AIAnalysis: React.FC<AIAnalysisProps> = ({ analysis }) => {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
        <Brain className="w-5 h-5 mr-2 text-purple-500" />
        AI Analysis (DeepSeek)
      </h2>

      <div className="mb-6 p-4 bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-lg">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm text-blue-600 font-medium">AI Recommendation</div>
            <div className={`text-2xl font-bold ${getTrendColor(analysis.recommendation)}`}>
              {analysis.recommendation}
            </div>
            <div className="text-sm text-gray-600 mt-1">
              Confidence: {(analysis.confidence_score * 100).toFixed(1)}%
            </div>
          </div>
          <div className={`px-4 py-2 rounded-full text-sm font-medium ${getRiskColor(analysis.risk_assessment)}`}>
            Risk: {analysis.risk_assessment}
          </div>
        </div>
      </div>

      <div className="space-y-4">
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2 flex items-center">
            <TrendingUp className="w-4 h-4 mr-2" />
            Market Sentiment
          </h3>
          <div className={`px-3 py-2 rounded-lg inline-block ${
            analysis.market_sentiment === 'BULLISH' 
              ? 'bg-green-100 text-green-800'
              : analysis.market_sentiment === 'BEARISH'
              ? 'bg-red-100 text-red-800'
              : 'bg-gray-100 text-gray-800'
          }`}>
            {analysis.market_sentiment}
          </div>
        </div>

        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2">Technical Summary</h3>
          <p className="text-gray-600 text-sm leading-relaxed">
            {analysis.technical_summary}
          </p>
        </div>

        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2">Fundamental Impact</h3>
          <p className="text-gray-600 text-sm leading-relaxed">
            {analysis.fundamental_impact}
          </p>
        </div>

        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2 flex items-center">
            <Target className="w-4 h-4 mr-2" />
            Price Prediction
          </h3>
          <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
            <p className="text-gray-700 text-sm mb-3">{analysis.price_prediction.short_term}</p>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <h4 className="text-xs font-medium text-green-600 mb-2">Support Targets</h4>
                <div className="space-y-1">
                  {analysis.price_prediction.targets.support.map((level, index) => (
                    <div key={index} className="text-sm text-gray-700">
                      ${level.toFixed(2)}
                    </div>
                  ))}
                </div>
              </div>
              
              <div>
                <h4 className="text-xs font-medium text-red-600 mb-2">Resistance Targets</h4>
                <div className="space-y-1">
                  {analysis.price_prediction.targets.resistance.map((level, index) => (
                    <div key={index} className="text-sm text-gray-700">
                      ${level.toFixed(2)}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2 flex items-center">
            <AlertTriangle className="w-4 h-4 mr-2" />
            Risk Assessment
          </h3>
          <div className={`px-3 py-2 rounded-lg inline-block ${getRiskColor(analysis.risk_assessment)}`}>
            {analysis.risk_assessment} RISK
          </div>
        </div>
      </div>
    </div>
  );
};

export default AIAnalysis;
