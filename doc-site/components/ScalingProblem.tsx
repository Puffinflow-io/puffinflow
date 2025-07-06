import React from 'react';
import { CheckIcon, CrossIcon, TrendingUpIcon, WrenchIcon, FeatherIcon } from './Icons';

const ScalingProblem: React.FC = () => {
  return (
    <section id="features" className="py-20 bg-[#1A1A1A]">
      <div className="container mx-auto px-6">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <h2 className="text-3xl md:text-4xl font-extrabold text-gray-50 tracking-tight">
            The Prototype-to-Production Problem
          </h2>
          <p className="mt-4 text-lg text-gray-300">
            Every AI framework gets you started fast. But Puffinflow is designed to scale with you, eliminating the need for costly rewrites when you hit production complexity.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16 max-w-6xl mx-auto">
          {/* Card 1: LangGraph/LlamaIndex */}
          <div className="border border-white/10 rounded-lg p-6 shadow-lg bg-gradient-to-b from-white/5 to-transparent flex flex-col transition-all duration-300 hover:border-white/20 hover:bg-white/10 lift-on-hover">
            <div className="flex items-center gap-3 mb-6">
              <TrendingUpIcon className="h-7 w-7 text-gray-400" />
              <h3 className="text-xl font-bold text-gray-100">LangGraph/LlamaIndex</h3>
            </div>
            <ul className="space-y-3 text-gray-300 flex-grow">
                <li className="flex items-start gap-3"><CheckIcon className="h-5 w-5 text-green-400 flex-shrink-0 mt-1" /><span>Great for rapid prototyping and demos.</span></li>
                <li className="flex items-start gap-3"><CrossIcon className="h-5 w-5 text-red-400 flex-shrink-0 mt-1" /><span>Hits a "production wall" requiring architectural changes.</span></li>
                <li className="flex items-start gap-3"><CrossIcon className="h-5 w-5 text-red-400 flex-shrink-0 mt-1" /><span>Reliability features must be added with external tools.</span></li>
            </ul>
          </div>

          {/* Card 2: Traditional Enterprise Tools */}
          <div className="border border-white/10 rounded-lg p-6 shadow-lg bg-gradient-to-b from-white/5 to-transparent flex flex-col transition-all duration-300 hover:border-white/20 hover:bg-white/10 lift-on-hover">
            <div className="flex items-center gap-3 mb-6">
              <WrenchIcon className="h-7 w-7 text-gray-400"/>
              <h3 className="text-xl font-bold text-gray-100">Traditional Tools</h3>
            </div>
            <ul className="space-y-3 text-gray-300 flex-grow">
                <li className="flex items-start gap-3"><CrossIcon className="h-5 w-5 text-red-400 flex-shrink-0 mt-1" /><span>Slow and complex setup, hindering initial speed.</span></li>
                <li className="flex items-start gap-3"><CheckIcon className="h-5 w-5 text-green-400 flex-shrink-0 mt-1" /><span>Built for enterprise-grade reliability from day one.</span></li>
                <li className="flex items-start gap-3"><CrossIcon className="h-5 w-5 text-red-400 flex-shrink-0 mt-1" /><span>High operational overhead, often over-engineered.</span></li>
            </ul>
          </div>

          {/* Card 3: Puffinflow */}
          <div className="border border-orange-500/40 rounded-lg p-6 shadow-xl shadow-orange-900/20 bg-gradient-to-br from-orange-900/20 to-orange-900/5 flex flex-col ring-1 ring-orange-500/40 transform md:scale-105 transition-all duration-300 hover:shadow-orange-600/40 hover:scale-[1.07]">
            <div className="flex items-center gap-3 mb-6">
              <FeatherIcon className="h-7 w-7 text-orange-400" />
              <h3 className="text-xl font-bold text-orange-400">Puffinflow</h3>
            </div>
             <ul className="space-y-3 text-gray-300 flex-grow">
                <li className="flex items-start gap-3"><CheckIcon className="h-5 w-5 text-green-400 flex-shrink-0 mt-1" /><span>Combines rapid prototyping with production-readiness.</span></li>
                <li className="flex items-start gap-3"><CheckIcon className="h-5 w-5 text-green-400 flex-shrink-0 mt-1" /><span>Add reliability features incrementally, without rewrites.</span></li>
                <li className="flex items-start gap-3"><CheckIcon className="h-5 w-5 text-green-400 flex-shrink-0 mt-1" /><span>Provides a smooth, single-codebase path to scale.</span></li>
            </ul>
          </div>
        </div>

         <div className="text-center max-w-3xl mx-auto mt-24">
            <h3 className="text-2xl font-bold text-gray-50 mb-4">Puffinflow Eliminates the Rewrite</h3>
            <p className="text-lg text-gray-300">
                Most teams choose frameworks based on demo speed, but the real cost is the inevitable rewrite for production. Puffinflow is designed to grow with you, saving you months of engineering effort.
            </p>
        </div>

      </div>
    </section>
  );
};

export default ScalingProblem;
