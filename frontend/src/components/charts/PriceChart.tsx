import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { format } from 'date-fns';

interface PriceChartProps {
  data: any[];
  supportLevels: number[];
  resistanceLevels: number[];
  height?: number;
}

const PriceChart: React.FC<PriceChartProps> = ({
  data,
  supportLevels,
  resistanceLevels,
  height = 400,
}) => {
  const formatDate = (timestamp: string) => {
    return format(new Date(timestamp), 'MMM dd');
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-300 rounded-lg shadow-lg">
          <p className="text-gray-600">{formatDate(label)}</p>
          <p className="text-gray-900 font-semibold">
            ${payload[0].value.toFixed(2)}
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis
            dataKey="timestamp"
            tickFormatter={formatDate}
            stroke="#6b7280"
            fontSize={12}
          />
          <YAxis
            stroke="#6b7280"
            fontSize={12}
            domain={['auto', 'auto']}
            tickFormatter={(value) => `$${value.toFixed(0)}`}
          />
          <Tooltip content={<CustomTooltip />} />
          
          {supportLevels.map((level, index) => (
            <ReferenceLine
              key={`support-${index}`}
              y={level}
              stroke="#22c55e"
              strokeDasharray="3 3"
              strokeWidth={1}
            />
          ))}
          
          {resistanceLevels.map((level, index) => (
            <ReferenceLine
              key={`resistance-${index}`}
              y={level}
              stroke="#ef4444"
              strokeDasharray="3 3"
              strokeWidth={1}
            />
          ))}
          
          <Line
            type="monotone"
            dataKey="close"
            stroke="#f59e0b"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: '#f59e0b' }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default PriceChart;
