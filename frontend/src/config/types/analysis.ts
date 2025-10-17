export interface PriceData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

export interface TechnicalIndicator {
  name: string;
  value: number;
  signal: 'BUY' | 'SELL' | 'NEUTRAL';
}

export interface TechnicalAnalysis {
  indicators: TechnicalIndicator[];
  summary: string;
  confidence: number;
  support_levels: number[];
  resistance_levels: number[];
}

export interface AIAnalysis {
  technical_summary: string;
  fundamental_impact: string;
  market_sentiment: string;
  price_prediction: {
    short_term: string;
    targets: {
      support: number[];
      resistance: number[];
    };
  };
  risk_assessment: string;
  recommendation: 'BUY' | 'SELL' | 'HOLD';
  confidence_score: number;
}

export interface AnalysisResponse {
  symbol: string;
  current_price: number;
  timestamp: string;
  technical_analysis: TechnicalAnalysis;
  ai_analysis: AIAnalysis;
  chart_data: any;
}

export interface MarketStatus {
  [timeframe: string]: {
    current_price: number;
    trend: string;
    confidence: number;
    support_levels: number[];
    resistance_levels: number[];
  };
}
