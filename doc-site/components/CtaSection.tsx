
import React from 'react';
import CodeBlock from './CodeBlock';

const CtaSection: React.FC = () => {
  const handleNavClick = (e: React.MouseEvent<HTMLAnchorElement>, hash: string) => {
    e.preventDefault();
    window.location.hash = hash;
  };

  return (
    <section className="py-20">
      <div className="container mx-auto px-6">
        <div className="relative bg-[#18181B] rounded-lg shadow-2xl p-8 md:p-12 text-center text-white border border-white/10 overflow-hidden">
          <div aria-hidden="true" className="absolute inset-0 -z-10">
              <div className="absolute inset-0 bg-gradient-to-br from-purple-600/10 to-transparent opacity-50" />
              <div className="absolute inset-x-0 bottom-0 h-1/2 bg-gradient-to-t from-[#18181B] to-transparent" />
          </div>
          <h2 className="text-3xl md:text-4xl font-extrabold tracking-tight text-gray-50">
            Ready for Reliable AI Workflows?
          </h2>
          <p className="mt-4 text-lg text-gray-300 max-w-2xl mx-auto">
            Stop debugging production failures. Start building reliable systems.
          </p>
          <div className="mt-8 max-w-sm mx-auto">
             <CodeBlock code="pip install puffinflow" language="bash" />
          </div>
          <div className="mt-8 flex justify-center items-center flex-wrap gap-4">
            <a
              href="#quickstart"
              onClick={(e) => handleNavClick(e, '#quickstart')}
              className="bg-gradient-to-r from-orange-500 to-orange-600 text-white px-6 py-3 rounded-md font-semibold hover:from-orange-600 hover:to-orange-700 transition-all duration-300 transform hover:scale-105 shadow-lg shadow-orange-600/20 hover:shadow-orange-500/40"
            >
              Get Started &rarr;
            </a>
            <a
              href="#docs"
              onClick={(e) => handleNavClick(e, '#docs')}
              className="bg-white/10 text-gray-200 px-6 py-3 rounded-md font-semibold hover:bg-white/20 transition-colors duration-200 border border-white/20 shadow-sm backdrop-blur-sm"
            >
              Documentation &rarr;
            </a>
          </div>
        </div>
      </div>
    </section>
  );
};

export default CtaSection;
