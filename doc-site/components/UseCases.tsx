import React from 'react';
import { CheckIcon, SparkleIcon } from './Icons';


const UseCases: React.FC = () => {
    const perfectFor = [
        "Orchestrating multi-step LLM chains with tight token budgets and API quotas",
        "Running hundreds of concurrent autonomous agents that coordinate through shared resources",
        "Needing exact resumption after interruption (cloud pre-emptible nodes, CI jobs)",
        "Requiring typed shared memory to avoid prompt-format drift between states",
    ];

    const greatFor = [
        "Complex agent workflows with dependencies and coordination",
        "Resource-constrained environments needing quota management",
        "Teams that want Airflow-like orchestration without the operational overhead",
        "Projects requiring deterministic, reproducible execution",
    ];

    const CaseList: React.FC<{title: string, items: string[], icon: React.ReactNode, borderColor: string}> = ({title, items, icon, borderColor}) => (
        <div className={`p-8 rounded-xl border ${borderColor} bg-white/5 shadow-lg backdrop-blur-sm transition-all duration-300 hover:border-white/40 hover:bg-white/10 lift-on-hover`}>
            <div className="flex items-center gap-3 mb-6">
                {icon}
                <h3 className="text-xl font-bold text-gray-100">{title}</h3>
            </div>
            <ul className="space-y-4">
                {items.map((item, index) => (
                    <li key={index} className="flex items-start gap-3">
                        <CheckIcon className="h-6 w-6 text-green-400 flex-shrink-0 mt-1" />
                        <span className="text-gray-300">{item}</span>
                    </li>
                ))}
            </ul>
        </div>
    );

    return (
        <section className="py-20 bg-black/20">
            <div className="container mx-auto px-6">
                <div className="text-center max-w-2xl mx-auto mb-12">
                    <h2 className="text-3xl md:text-4xl font-extrabold text-gray-50 tracking-tight">
                        When to Choose Puffinflow
                    </h2>
                     <p className="mt-4 text-lg text-gray-400">
                        Puffinflow is versatile, but it truly excels where reliability is non-negotiable.
                    </p>
                </div>
                <div className="grid md:grid-cols-2 gap-8 max-w-5xl mx-auto">
                    <CaseList
                        title="Perfect For..."
                        items={perfectFor}
                        icon={<CheckIcon className="h-7 w-7 text-green-400" />}
                        borderColor="border-green-500/20"
                    />
                    <CaseList
                        title="Great For..."
                        items={greatFor}
                        icon={<SparkleIcon className="h-7 w-7 text-amber-400" />}
                        borderColor="border-amber-500/20"
                    />
                </div>
            </div>
        </section>
    );
};

export default UseCases;
