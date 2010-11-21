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

    Stores the values of the trained naive bayes classifier.

    @author Marek Schmidt
*/
public class NaiveBayesClassifier implements Serializable {

        private static final long serialVersionUID = -6420394538829715510L;
    
        public Map<Integer, Double> logpriors;
        public Map<Integer, Map<String, Double>> loglikelihoods;
        public Map<Integer, List<String>> features;
    
        public NaiveBayesClassifier() {
            logpriors = new HashMap<Integer, Double> ();
            loglikelihoods = new HashMap<Integer, Map<String, Double>> ();
            features = new HashMap<Integer, List<String>> ();
        }
    
        public String toString() {
            StringBuilder sb = new StringBuilder();

            sb.append("NBC: \n");

            for (Map.Entry<Integer, List<String>> entry : features.entrySet()) {
                sb.append("" + entry.getKey() + ": ");
                for (String feature : entry.getValue()) {
                    sb.append("" + feature + "\n");
                }
                sb.append("\n");
            }


            for (Map.Entry<Integer, Map<String, Double>> entry : loglikelihoods.entrySet()) {
                sb.append("" + entry.getKey() + ": ");
                for (Map.Entry<String, Double> entry2 : entry.getValue().entrySet()) {
                    sb.append("" + entry2.getKey() + "/" + entry2.getValue() + " ");
                }
                sb.append("\n");
            }

            return sb.toString();
        }
}

