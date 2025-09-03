import React from 'react';
import { 
    ChainBrokenIcon, 
    MemoryLeakIcon, 
    RaceConditionIcon, 
    NoVisibilityIcon,
    CircuitBreakerIcon,
    ManagedResourcesIcon,
    StateGuardIcon,
    FullObservabilityIcon,
    AlertIcon,
    ShieldCheckIcon
} from './Icons';

const ProductionGap: React.FC = () => {
    const problems = [
        { text: "API failures cascade through pipelines", icon: ChainBrokenIcon },
        { text: "Memory leaks crash long-running jobs", icon: MemoryLeakIcon },
        { text: "Race conditions corrupt state", icon: RaceConditionIcon },
        { text: "No visibility into what actually failed", icon: NoVisibilityIcon },
    ];
     const solutions = [
        { text: "Circuit breakers prevent cascading failures", icon: CircuitBreakerIcon },
        { text: "Managed resources stop memory leaks", icon: ManagedResourcesIcon },
        { text: "State guards prevent race conditions", icon: StateGuardIcon },
        { text: "Full observability into every step", icon: FullObservabilityIcon },
    ];

    return (
        <section className="py-20">
            <div className="container mx-auto px-6">
                <div className="text-center max-w-3xl mx-auto">
                    <h2 className="text-3xl md:text-4xl font-extrabold text-gray-50 tracking-tight">
                        From Prototype Mess to Production Success
                    </h2>
                    <p className="mt-4 text-lg text-gray-300">
                       Most AI frameworks get you a demo. Puffinflow gets you a product.
                    </p>
                </div>
                <div className="mt-12 grid grid-cols-1 md:grid-cols-2 gap-8 max-w-5xl mx-auto">
                   {/* The Problem */}
                   <div className="bg-red-900/10 border border-red-500/20 rounded-xl p-8 shadow-2xl shadow-red-900/10 lift-on-hover transition-all duration-300 hover:border-red-500/40">
                       <div className="flex items-center gap-4 mb-6">
                           <AlertIcon className="h-8 w-8 text-red-400 flex-shrink-0" />
                           <h3 className="text-2xl font-bold text-red-300">The Tangled Reality</h3>
                       </div>
                       <ul className="space-y-4">
                           {problems.map((problem, index) => {
                               const IconComponent = problem.icon;
                               return (
                                   <li key={index} className="flex items-start gap-3">
                                       <IconComponent className="h-5 w-5 text-red-500/70 flex-shrink-0 mt-1" />
                                       <p className="text-gray-300 font-medium">{problem.text}</p>
                                   </li>
                               );
                           })}
                       </ul>
                   </div>
                   {/* The Solution */}
                   <div className="bg-green-900/10 border border-green-500/20 rounded-xl p-8 shadow-2xl shadow-green-900/10 lift-on-hover transition-all duration-300 hover:border-green-500/40">
                       <div className="flex items-center gap-4 mb-6">
                           <ShieldCheckIcon className="h-8 w-8 text-green-400 flex-shrink-0" />
                           <h3 className="text-2xl font-bold text-green-300">The Puffinflow Promise</h3>
                       </div>
                        <ul className="space-y-4">
                           {solutions.map((solution, index) => {
                               const IconComponent = solution.icon;
                               return (
                                   <li key={index} className="flex items-start gap-3">
                                       <IconComponent className="h-5 w-5 text-green-500/80 flex-shrink-0 mt-1" />
                                       <p className="text-gray-200 font-medium">{solution.text}</p>
                                   </li>
                               );
                           })}
                       </ul>
                   </div>
                </div>
                 <div className="mt-16 text-center max-w-3xl mx-auto">
                    <p className="text-xl md:text-2xl font-bold text-gray-100">
                        Stop patching broken workflows. <br/>
                        <span className="text-orange-400">Build them right from the start.</span>
                    </p>
                </div>
            </div>
        </section>
    );
};

export default ProductionGap;
