import React from 'react';
import { formatCurrency } from '../../utils/formatting';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface PriceDisplayProps {
  currentPrice: number;
  previousPrice?: number;
  size?: 'sm' | 'md' | 'lg';
}

const PriceDisplay: React.FC<PriceDisplayProps> = ({
  currentPrice,
  previousPrice,
  size = 'md',
}) => {
  const getChange = () => {
    if (!previousPrice) return null;
    const change = currentPrice - previousPrice;
    const changePercent = (change / previousPrice) * 100;
    return { change, changePercent };
  };

  const changeData = getChange();
  const isPositive = changeData && changeData.change > 0;
  const isNegative = changeData && changeData.change < 0;

  const sizeClasses = {
    sm: 'text-2xl',
    md: 'text-3xl',
    lg: 'text-4xl'
  };

  return (
    <div className="flex items-end space-x-3">
      <div className={`font-bold text-gray-900 ${sizeClasses[size]}`}>
        {formatCurrency(currentPrice)}
      </div>
      
      {changeData && (
        <div className={`flex items-center space-x-1 text-sm font-medium mb-1 ${
          isPositive ? 'text-green-600' : isNegative ? 'text-red-600' : 'text-gray-600'
        }`}>
          {isPositive && <TrendingUp className="w-4 h-4" />}
          {isNegative && <TrendingDown className="w-4 h-4" />}
          {!isPositive && !isNegative && <Minus className="w-4 h-4" />}
          
          <span>
            {isPositive ? '+' : ''}{formatCurrency(changeData.change)} 
            ({isPositive ? '+' : ''}{changeData.changePercent.toFixed(2)}%)
          </span>
        </div>
      )}
    </div>
  );
};

export default PriceDisplay;
