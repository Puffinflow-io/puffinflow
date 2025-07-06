import React from 'react';
import { CheckIcon, CrossIcon } from './Icons';

const Comparison: React.FC = () => {
  const data = [
    { feature: 'Circuit Breakers', puffin: <CheckIcon />, lang: <CrossIcon />, llama: <CrossIcon /> },
    { feature: 'Resource Management', puffin: <CheckIcon />, lang: <CrossIcon />, llama: <CrossIcon /> },
    { feature: 'Deadlock Detection', puffin: <CheckIcon />, lang: <CrossIcon />, llama: <CrossIcon /> },
    { feature: 'Memory Leak Detection', puffin: <CheckIcon />, lang: <CrossIcon />, llama: <CrossIcon /> },
  ];

  const ToolHeader: React.FC<{name: string, highlight?: boolean}> = ({ name, highlight }) => (
    <th scope="col" className={`px-6 py-4 text-center font-bold tracking-wider ${highlight ? 'text-orange-400' : 'text-gray-200'}`}>
      {name}
    </th>
  );

  return (
    <section id="compare" className="py-20 bg-black/20">
      <div className="container mx-auto px-6">
        <div className="text-center max-w-2xl mx-auto mb-12">
          <h2 className="text-3xl md:text-4xl font-extrabold text-gray-50 tracking-tight">
            Choose the Right Tool
          </h2>
          <p className="mt-4 text-lg text-gray-400">A feature-level look at where Puffinflow shines for production needs.</p>
        </div>
        <div className="max-w-4xl mx-auto">
          <div className="overflow-x-auto rounded-lg border border-white/10 shadow-xl bg-gradient-to-b from-[#1A1A1A]/80 to-[#2A2A2E]/50 backdrop-blur-sm">
            <table className="w-full text-sm text-left text-gray-300">
              <thead className="text-sm text-gray-400 uppercase bg-white/5">
                <tr>
                  <th scope="col" className="px-6 py-4 font-bold text-gray-100 tracking-wider">
                    Production Feature
                  </th>
                  <ToolHeader name="Puffinflow" highlight />
                  <ToolHeader name="LangGraph" />
                  <ToolHeader name="LlamaIndex" />
                </tr>
              </thead>
              <tbody>
                {data.map((row, index) => (
                  <tr key={index} className="border-b border-white/10 last:border-b-0 hover:bg-white/5 comparison-table-row">
                    <th scope="row" className="px-6 py-5 font-bold text-gray-100 whitespace-nowrap">
                      {row.feature}
                    </th>
                    <td className="px-6 py-4 text-center"><div className="flex justify-center">{row.puffin}</div></td>
                    <td className="px-6 py-4 text-center"><div className="flex justify-center">{row.lang}</div></td>
                    <td className="px-6 py-4 text-center"><div className="flex justify-center">{row.llama}</div></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Comparison;
