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

package cz.vutbr.fit.nlp.lc.tools;

import cz.vutbr.fit.nlp.lc.*;

import java.io.Serializable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
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
 * An example how to use the library. Provide the list of positive and
 * negative examples as training data, the tool will output the top most 20 
 * likely documents that belong to the class.
 *
 * The input may look like this:
 *
 * {@code
 * +42
 * +128
 * -74
 * -80
 * }
 *
 * Where the numbers are the "id" fields of the indexed documents, whatever
 * they are (they don't have to be numbers)
 *
 * To run, you also need to specify path to the index:
 *
 * {@code java -cp lucene-core-2.9.1.jar:. cz.vutbr.fit.nlp.lc.tools.Search index/}
 * 
 * @author Marek Schmidt
 */
class Search {

    public static class Result {
        public double score;
        public int docn;

        public Result(int docn, double score) {
            this.docn = docn;
            this.score = score;
        }
    }

    public static void main(String[] args) throws Exception {

        IndexReader reader;
        
        Directory dir = FSDirectory.getDirectory(new File(args[0]));
                
        reader = IndexReader.open(dir);

        List<String> poses = new LinkedList<String>();
        List<String> neges = new LinkedList<String>();

        Map<Integer, Set<Integer>> class2ids = new HashMap<Integer, Set<Integer>>();

        class2ids.put(0, new HashSet<Integer>());
        class2ids.put(1, new HashSet<Integer>());

        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, "UTF-8"));
        String line;
        while( (line = in.readLine()) != null) {
            boolean pos;
            if ("+".equals(line.substring(0, 1))) {
                pos = true;
            }
            else if ("-".equals(line.substring(0, 1))) {
                pos = false;
            }
            else {
                continue;
            }

            String title = line.substring(1);
            Term term = new Term("id", title);
            TermDocs docs = reader.termDocs(term);
            docs.next();
            int docid = docs.doc();

            System.err.println("term " + title + ": docid: " + docid);

            if (pos) {
                poses.add (title);
                class2ids.get(0).add(docid);
            }
            else {
                neges.add (title);
                class2ids.get(1).add(docid);
            }
        }

        NaiveBayesClassifier nbc;
        nbc = LuceneClassification.learn(reader, class2ids, "lemmas");

        // System.out.println(nbc.toString());
        
        LuceneClassification classification = new LuceneClassification(reader, nbc, "lemmas", 0);
        LuceneClassification.Iteration iter = null;// classification.steps(null, 50);

        while(classification.hasNext(iter)) {
            iter = classification.step(iter);

            System.err.println("Iteration: " + iter.getIteration() + ", feature: " + classification.getFeatures().get(iter.getIteration()) + " ll: " + nbc.loglikelihoods.get(0).get(classification.getFeatures().get(iter.getIteration())));

            // We fix the number of features to 50, which has quite acceptable
            // results in both speed and classification performance.
            if (iter.getIteration() >= 50) break;
        }


        PriorityQueue<Result> pq = new PriorityQueue(20, new Comparator<Result>() {
            public int compare(Result r1, Result r2) {
                return Double.compare(r1.score, r2.score);
            }
        });

        for (Map.Entry<Integer, Double> entry : iter.getId2LogScore().entrySet()) {

            if (class2ids.get(0).contains(entry.getKey()) || class2ids.get(1).contains(entry.getKey())) {
                continue;
            }

            //Document doc = reader.document(entry.getKey());
            //Field field = doc.getField("id");
                //TokenStream stream = field.tokenStreamValue();
            //String id = field.stringValue();

            if (pq.size() < 20) {
                pq.add(new Result(entry.getKey(), entry.getValue()));
            }
            else {
                if (pq.peek().score < entry.getValue()) {
                    pq.poll();
                    pq.add(new Result(entry.getKey(), entry.getValue()));
                }
            }
        }

        // note, this will print the best results in wrong order... 
        while (pq.size() > 0) {
            Result r = pq.poll();

            Document doc = reader.document(r.docn);
            Field field = doc.getField("id");
            // TokenStream stream = field.tokenStreamValue();
            String id = field.stringValue();

            field = doc.getField("lemmas");
            //stream = field.tokenStreamValue();
            String lemmas = field.stringValue();

            field = doc.getField("title");
            String title = "";
            if (field != null) {
                title = field.stringValue();
            }

            if (lemmas.length() > 76) {
                lemmas = lemmas.substring(0, 75) + "...";
            }

            System.out.println("" + r.score + "\t" + id + "\t" + title + "\t" + lemmas);
        }
        
        reader.close();
    }
}

