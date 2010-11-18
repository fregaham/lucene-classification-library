/*
 * Copyright (c) 2010, Marek Schmidt
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are met:
 * - Redistributions of source code must retain the above copyright notice, 
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice, 
 *   this list of conditions and the following disclaimer in the documentation 
 *   and/or other materials provided with the distribution.
 * - Neither the name of the Brno University of Technology nor the names of its
 *   contributors may be used to endorse or promote products derived from this 
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 * POSSIBILITY OF SUCH DAMAGE.
 * 
 * Contributor(s):
 *     Marek Schmidt <fregaham@gmail.com>
 * 
 */

package cz.vutbr.fit.nlp.lc;

import java.io.Serializable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;

import java.io.File;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import org.apache.lucene.analysis.WhitespaceAnalyzer;
import org.apache.lucene.analysis.Token;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TermDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

/**
    The core class of the lucene classification library.

    It 

    @author Marek Schmidt
*/
public class LuceneClassification {

    public static class Iteration {
        Map<Integer, Double> id2logscore;
        int iteration;
    }

    private IndexReader reader;
    private NaiveBayesClassifier classifier;
    private String featureField;
    private Map<String, Double> loglikelihoods;
    private List<String> features;
    private double logprior;
    private int klass;
    
    public LuceneClassification(IndexReader reader, NaiveBayesClassifier classifier, String featureField, int klass) {
         this.reader = reader;
         this.classifier = classifier;
         this.featureField = featureField;
         this.klass = klass;
         
         loglikelihoods = classifier.loglikelihoods.get(klass);
         logprior = classifier.logpriors.get(klass);
         features = classifier.features.get(klass);
    }
    
    public boolean hasNext(Iteration prev) {
        return prev == null ? features.size() > 0 : features.size() > prev.iteration + 1;
    }

    public List<String> getFeatures() {
        return features;
    }

    public Iteration steps(Iteration prev, int n) throws IOException {
        while (hasNext(prev) && n > 0) {
            prev = step(prev);
            n--;
        }

        return prev;
    }
        
    public Iteration step(Iteration prev) throws IOException {
        
        if (!hasNext(prev)) return prev;
        
        int currentTermIndex = prev == null ? 0 : prev.iteration + 1;
        String currentTerm = features.get(currentTermIndex);
        
        Map<Integer, Double> prevId2logscore = prev == null ? new HashMap<Integer, Double>() : prev.id2logscore;
        Map<Integer, Double> id2logscore = prevId2logscore;
        
        Term term = new Term(this.featureField, features.get(currentTermIndex));
        
        TermDocs tds = reader.termDocs(term);
        while(tds.next()) {
            int docid = tds.doc();
            
            Double prevValueObj = prevId2logscore.get(docid);
            double prevValue;
            if (prevValueObj == null) {
                // Usually it doesn't make much sense to care about priors here...

                //prevValue = logprior;
                prevValue = 0.0;
            }
            else {
                prevValue = prevValueObj;
            }

            
            double logLikelihood = 0.0;
            logLikelihood = loglikelihoods.get(currentTerm);
            id2logscore.put(docid, prevValue + logLikelihood);
        }
        
        Iteration next = new Iteration();
        if (prev == null) {
            next.iteration = 1;
        }
        else {
            next.iteration = prev.iteration + 1;
        }
        next.id2logscore = id2logscore;
        
        return next;
    }

    /**
        Classify the ordinary way (document to score). The positive value means the document does belong to the class. 
        @param nbc The trained classifier
        @param features The document to classify, represented as features.
        @param klass The klass we want to know if the document belongs to it or not.
        @param selectedfeaturs The number of features to use in the classification. -1 for all features, otherwise, only the best-selectedfeaturs features will be used.
        @return Positive value if the document does belong to the klass. 
    */
    public static double classify(NaiveBayesClassifier nbc, Collection<String> features, int klass, int selectedfeatures) {
        // Ignore log priors.
        // double p = logpriors.get(klass);
        double p = 0.0;

        if (selectedfeatures < 0) {
            Map<String, Double> klassloglikelihoods = nbc.loglikelihoods.get(klass);
            for (String f : features) {
                if (klassloglikelihoods.containsKey(f)) {
                    p += klassloglikelihoods.get(f);
                }
                // ignore features not encountered during training
            }
        }
        else {
            List<String> nbcfeatures = nbc.features.get(klass);

            Set<String> sfeatures = new HashSet<String>();
            for (int i = 0; i < selectedfeatures; ++i) {
                sfeatures.add (nbcfeatures.get(i));
            }

            Map<String, Double> klassloglikelihoods = nbc.loglikelihoods.get(klass);
            for (String f : features) {
                if (sfeatures.contains(f) && klassloglikelihoods.containsKey(f)) {
                    p += klassloglikelihoods.get(f);
                }
                // ignore features not encountered during training
            }

        }
        
        return p;
    }

    public static double classify(NaiveBayesClassifier nbc, Collection<String> features, int klass) {
        return classify(nbc, features, klass, -1);
    }
    
    public static double classify(NaiveBayesClassifier nbc, IndexReader reader, int docid, String featureField, int klass, int selectedfeatures) throws IOException {
        Document doc = reader.document(docid);
        Field field = doc.getField(featureField);
        String stringvalue = field.stringValue();
        String[] tokens = stringvalue.split(" ");
        return classify(nbc, Arrays.asList(tokens), klass, selectedfeatures);
    }

    public static double classify(NaiveBayesClassifier nbc, IndexReader reader, int docid, String featureField, int klass) {
        return classify(nbc, reader, docid, featureField, -1);
    }

    /**
        a function for likelihood ratio computation, doesn't mean anything by itself...
    */ 
    private static double l(double n, double N, double pc, double pj) {                                                                                                                                                                   
        return n * (Math.log(pc) - Math.log(pj)) + (N - n) * (Math.log(1.0 - pc / pj));   
    }

    /**
        a function to make the entropy calculation formula for mutual information more managable... 
    */
    private static double F(double x, double T) {
        return x * Math.log(x) / T;
    }

    public static NaiveBayesClassifier learn(IndexReader reader, Map<Integer, Set<Integer>> class2ids, String featureField) throws IOException {
        NaiveBayesClassifier ret = new NaiveBayesClassifier();
        
        int total = 0;

        // for each class, we store number of positive and negative occurences of a term.
        // Positive means the term occured in the document positively labelled by the class label...

        Map<Integer, Map<String, Integer>> pos = new HashMap<Integer, Map<String, Integer>> ();
        Map<Integer, Map<String, Integer>> neg = new HashMap<Integer, Map<String, Integer>> ();
 
        for (Integer klass : class2ids.keySet()) {
            pos.put(klass, new HashMap<String,Integer> ());
            neg.put(klass, new HashMap<String,Integer> ());
        }
        
        for (Integer klass : class2ids.keySet()) {
            total += class2ids.get(klass).size();
            
            for (Integer docid : class2ids.get(klass)) {

                Document doc = reader.document(docid);
                Field field = doc.getField(featureField);
                String stringvalue = field.stringValue();
                String[] tokens = stringvalue.split(" ");
                for(String text : tokens) {    
                      for (Integer klass2 : class2ids.keySet()) {
                        if (klass.equals(klass2)) {
                            if (pos.get(klass2).containsKey(text)) {
                                int value = pos.get(klass2).get(text);
                                pos.get(klass2).put(text, 1 + value);
                            }
                            else {
                                pos.get(klass2).put(text, 1);
                            }
                        }
                        else {
                            if (neg.get(klass2).containsKey(text)) {
                                int value = neg.get(klass2).get(text);
                                neg.get(klass2).put(text, value + 1);
                            }
                            else {
                                neg.get(klass2).put(text, 1);
                            }
                        }
                    }
                }
            }
        }
        
        ret.logpriors = new HashMap<Integer, Double> ();
        ret.loglikelihoods = new HashMap<Integer, Map<String, Double>>();
        ret.features = new HashMap<Integer, List<String>> ();
        for (Integer klass : class2ids.keySet()) {
            
            ret.logpriors.put(klass, Math.log(class2ids.get(klass).size()) - Math.log(total - class2ids.get(klass).size()));
            
            Map<String, Double> klassloglikelihoods = new HashMap<String, Double>(); 
            final Map<String, Double> miscores = new HashMap<String, Double>();
            Map<String, Double> fscores = new HashMap<String, Double>();

            for (Map.Entry<String, Integer> pose : pos.get(klass).entrySet()) {
                String term = pose.getKey();

                if (term.length() < 3) {
                    continue;
                }

                double posValue = pose.getValue();
                double negValue;
                if (neg.get(klass).containsKey(pose.getKey())) {
                    negValue = neg.get(klass).get(pose.getKey());
                }
                else {
                    // add one smooth
                    negValue = 1.0;
                }

                // The following code is garbage, because that's where I play with different feature selection schemes
                // The methods you can find here: Information gain, mutual information, likelihood ratio, and some variants...
 
                //double loglikelihood = Math.log(posValue) - Math.log(negValue);//Math.log(posValue + negValue);
                double loglikelihood = Math.log(posValue) - Math.log(class2ids.get(klass).size()) - Math.log(negValue) + Math.log(0.0 + total - class2ids.get(klass).size());
                                
/*                double miA = posValue;
                double miB = negValue;
                double miC = class2ids.get(klass).size() - posValue;
                double miN = total;
                double mi = miA * miN / ((miA + miC) * (miA + miB));*/

//                double featurescore = posValue + negValue;

                double Pc = (0.0 + class2ids.get(klass).size()) / (0.0 + total);
                double Pnc = 1.0 - Pc;
                double Pt = (0.0 + posValue + negValue) / (0.0 + total);
                double Pnt = 1.0 - Pt;
                double Pc_t = (0.0 + posValue) / (0.0 + posValue + negValue);
                double Pc_nt = (0.0 + class2ids.get(klass).size() - posValue) / (0.0 + total - (posValue + negValue));
                double Pnc_t = (0.0 + negValue) / (0.0 + posValue + negValue);
                double Pnc_nt = (0.0 + total - (class2ids.get(klass).size() + negValue)) / (0.0 + total - (posValue + negValue));

                double Ptc = (0.0 + posValue) / (0.0 + total);
                double Ptnc = (0.0 + negValue) / (0.0 + total);
                double Pntc = (0.0 + class2ids.get(klass).size() - posValue) / (0.0 + total);
                double Pntnc = (0.0 + total - (class2ids.get(klass).size() + negValue)) / (0.0 + total);

                double ig = - Pc * Math.log(Pc) - Pnc * Math.log(Pnc) + Pt * (Pc_t * Math.log(Pc_t) + Pnc_t * Math.log(Pnc_t)) + Pnt * (Pc_nt * Math.log(Pc_nt) + Pnc_nt * Math.log(Pnc_nt));
                double score = ig;

                // estimate Pt from the whoe collection:
                //Pt = (0.0 + reader.docFreq(new Term(featureField, term))) / (0.0 + reader.numDocs());
//                System.err.println("Pt = " + Pt + " docFreq: " + reader.docFreq(new Term(featureField, term))+ " numDocs: " + reader.numDocs());

                /*if (loglikelihood <= 0) {
                    continue;
                }*/

                // same as ig
                /*double mi = Ptc   * Math.log( Ptc   / (Pc  * Pt)) + 
                            Ptnc  * Math.log( Ptnc  / (Pnc * Pt)) + 
                            Pntc  * Math.log( Pntc  / (Pc  * Pnt))+
                            Pntnc * Math.log( Pntnc / (Pnc * Pnt));*/

                //double mi = Ptc * Math.log(Ptc / (Pc * Pt)) + Ptnc  * Math.log( Ptnc  / (Pnc * Pt));

                //double mi = Pc * Math.log( Ptc / (Pc  * Pt)) + Pnc * Math.log( Ptnc  / (Pnc * Pt));
//                double mi = Math.max(Pc * Math.log( Ptc / (Pc  * Pt)), Pnc * Math.log( Ptnc  / (Pnc * Pt)));

                // MI1
                //double mi = Math.log(Ptc) - (Math.log(Pc) + Math.log(Pt));

                // MI2
                //double mi = Ptc * (Math.log(Ptc) - (Math.log(Pc) + Math.log(Pt))) + Ptnc * (Math.log(Ptnc) - (Math.log(Pnc) + Math.log(Pt)));


                double T = total;
                double Nc = class2ids.get(klass).size();
                double Nnc = T - Nc;

                double Nt = posValue + negValue;
                double Nnt = T - Nt;

                double Ntc = posValue;
                double Ntnc = negValue;
                double Nntc = Nc - Ntc;
                double Nntnc = Nnc - Ntnc;

                // MI3 (should be the "true" mutual information)
                /*double score = (Math.log(T) - (F(Nc, T) + F(Nnc, T)))  // H(Y)
                     + (Math.log(T) - (F(Nt, T) + F(Nnt, T))) // H(Xt)
                     + (Math.log(T) - (F(Ntc, T) + F(Ntnc, T) + F(Nntc, T) + F(Nntnc, T))); // H(Y, Xt) */

                // DF
                // double score = (Math.log(reader.docFreq(new Term(featureField, term))) - reader.numDocs());

                // LL
                //double score = loglikelihood;

                // LR1 / 2 / 3   (variants for likelihood ratio)
                double n = posValue;
                //double n_ = negVaue;
                //double n_ = 0.0 + reader.docFreq(new Term(featureField, term));
                double n_ = posValue + negValue;

                double N = class2ids.get(klass).size();
                //double N_ = total - class2ids.get(klass).size();
                //double N_ = reader.numDocs();
                double N_ = total;
                //double score = l(n_, N_, n + n_, N + N_) + l(n, N, n + n_, N + N_) - l(n, N, n, N) - l(n_, N_, n_, N_);

                fscores.put(pose.getKey(), score);

                klassloglikelihoods.put(pose.getKey(), loglikelihood);
            }
            
            ret.loglikelihoods.put(klass, klassloglikelihoods);
            
            List<Map.Entry<String, Double>> list = new ArrayList<Map.Entry<String, Double>>(fscores.entrySet());

            Collections.sort(list, new Comparator<Map.Entry<String, Double>>() {
                @Override
                public int compare(Entry<String, Double> o1,
                        Entry<String, Double> o2) {
                    return - Double.compare(o1.getValue(), o2.getValue());
                }
            });
            
            List<String> features = new LinkedList<String> ();
            for (Map.Entry<String, Double> entry : list) {
                features.add(entry.getKey());
            }
            
            ret.features.put(klass, features);
        }
        
        return ret;
    }

}
