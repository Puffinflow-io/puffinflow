
import React, { useState, useEffect, useRef } from 'react';
import Header from './components/Header';
import Hero from './components/Hero';
import ProductionGap from './components/ProductionGap';
import Comparison from './components/Comparison';
import ScalingProblem from './components/ScalingProblem';
import WhyPuffinflow from './components/WhyPuffinflow';
import QuickStart from './components/QuickStart';
import CtaSection from './components/CtaSection';
import Footer from './components/Footer';
import UseCases from './components/UseCases';
import { DocsPage, GettingStartedPage, ContextAndDataPage, ResourceManagementPage, ErrorHandlingPage, CheckpointingPage, RAGRecipePage } from './components/DocsPage';

const AnimatedSection: React.FC<{ children: React.ReactNode, animationType?: string }> = ({ children, animationType = 'reveal-up' }) => {
    const ref = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const element = ref.current;
        if (!element) return;

        const observer = new IntersectionObserver(([entry]) => {
            if (entry.isIntersecting) {
                element.classList.add('is-visible');
                observer.unobserve(element);
            }
        }, {
            threshold: 0.1,
        });

        observer.observe(element);

        return () => {
            if (element) {
                observer.unobserve(element);
            }
        };
    }, []);

    return (
        <div ref={ref} className={`reveal ${animationType}`}>
            {children}
        </div>
    );
};

const getRouteInfo = () => {
    const hash = window.location.hash.slice(1);

    if (hash.startsWith('docs/getting-started')) {
        const [, anchor] = hash.split('#');
        return { path: '/docs/getting-started', anchor: anchor || null };
    }
    if (hash.startsWith('docs/context-and-data')) {
        const [, anchor] = hash.split('#');
        return { path: '/docs/context-and-data', anchor: anchor || null };
    }
    if (hash.startsWith('docs/resource-management')) {
        const [, anchor] = hash.split('#');
        return { path: '/docs/resource-management', anchor: anchor || null };
    }
    if (hash.startsWith('docs/error-handling')) {
        const [, anchor] = hash.split('#');
        return { path: '/docs/error-handling', anchor: anchor || null };
    }
    if (hash.startsWith('docs/checkpointing')) {
        const [, anchor] = hash.split('#');
        return { path: '/docs/checkpointing', anchor: anchor || null };
    }
    if (hash.startsWith('docs/rag-recipe')) {
        const [, anchor] = hash.split('#');
        return { path: '/docs/rag-recipe', anchor: anchor || null };
    }
    if (hash.startsWith('docs')) {
        const [, anchor] = hash.split('#');
        return { path: '/docs', anchor: anchor || null };
    }

    const mainPageAnchor = hash;
    return { path: '/', anchor: mainPageAnchor || null };
};


const App: React.FC = () => {
  const [routeInfo, setRouteInfo] = useState(getRouteInfo());

  useEffect(() => {
      const handleHashChange = () => {
          setRouteInfo(getRouteInfo());
      };
      window.addEventListener('hashchange', handleHashChange);
      return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  // Scrolling effect
  useEffect(() => {
      if (routeInfo.anchor) {
          setTimeout(() => {
              const element = document.getElementById(routeInfo.anchor);
              if (element) {
                  element.scrollIntoView({ behavior: 'smooth' });
              }
          }, 100); // Timeout to allow page to render before scrolling
      } else if (routeInfo.path !== '/') {
          window.scrollTo(0, 0);
      }
  }, [routeInfo]);

  // Spotlight & animated background effect
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
        const { clientX, clientY } = e;
        const { innerWidth, innerHeight } = window;

        // Spotlight effect
        document.documentElement.style.setProperty('--mouse-x', `${clientX}px`);
        document.documentElement.style.setProperty('--mouse-y', `${clientY}px`);

        // Background grid rotation for 3D effect
        const rotateX = -((clientY / innerHeight) - 0.5) * 15; // Range: -7.5deg to 7.5deg
        const rotateY = ((clientX / innerWidth) - 0.5) * 15;  // Range: -7.5deg to 7.5deg

        document.documentElement.style.setProperty('--bg-rotate-x', `${rotateX}deg`);
        document.documentElement.style.setProperty('--bg-rotate-y', `${rotateY}deg`);
    };

    window.addEventListener('mousemove', handleMouseMove);

    return () => {
        window.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);


  const mainPageSections = [
    { Component: Hero, animation: 'reveal-zoom' },
    { Component: ProductionGap, animation: 'reveal-up' },
    { Component: WhyPuffinflow, animation: 'reveal-up' },
    { Component: ScalingProblem, animation: 'reveal-zoom' },
    { Component: Comparison, animation: 'reveal-up' },
    { Component: UseCases, animation: 'reveal-up' },
    { Component: QuickStart, animation: 'reveal-up' },
    { Component: CtaSection, animation: 'reveal-zoom' },
  ];

  const MainPage = () => (
     <>
      {mainPageSections.map(({ Component, animation }, index) => (
        <AnimatedSection key={index} animationType={animation}>
          <Component />
        </AnimatedSection>
      ))}
    </>
  );

  const renderPage = () => {
      switch (routeInfo.path) {
        case '/docs':
          return <DocsPage />;
        case '/docs/getting-started':
          return <GettingStartedPage />;
        case '/docs/context-and-data':
          return <ContextAndDataPage />;
        case '/docs/resource-management':
            return <ResourceManagementPage />;
        case '/docs/error-handling':
            return <ErrorHandlingPage />;
        case '/docs/checkpointing':
            return <CheckpointingPage />;
        case '/docs/rag-recipe':
            return <RAGRecipePage />;
        default:
          return <MainPage />;
      }
  };

  return (
    <div className="antialiased">
      <Header />
      <main>
        {renderPage()}
      </main>
      <Footer />
    </div>
  );
};

export default App;
