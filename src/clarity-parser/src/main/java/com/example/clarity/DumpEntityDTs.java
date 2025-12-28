package com.example.clarity;

import skadistats.clarity.model.Entity;
import skadistats.clarity.processor.entities.Entities;
import skadistats.clarity.processor.runner.Context;
import skadistats.clarity.processor.runner.SimpleRunner;
import skadistats.clarity.source.MappedFileSource;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

/**
 * Small helper: iterate entities and print distinct dt names and counts.
 * Usage:
 *   mvn -DskipTests clean package
 *   /usr/bin/java -cp target/clarity-parser-0.1.0-with-deps.jar com.example.clarity.DumpEntityDTs <replay.dem> <max_entities_to_scan>
 */
public class DumpEntityDTs {

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: DumpEntityDTs <replay.dem> <max_entities_to_scan>");
            System.exit(2);
        }
        String dem = args[0];
        int maxScan = Integer.parseInt(args[1]);

        SimpleRunner runner = new SimpleRunner(new MappedFileSource(dem));
        // Use an anonymous processor to run once and inspect the Entities processor via Context
        runner.runWith(new Object() {
            public void init(Context ctx) {
                Entities ents = ctx.getProcessor(Entities.class);
                if (ents == null) {
                    System.err.println("Entities processor not available.");
                    return;
                }
                Iterator<Entity> it = ents.getAllByPredicate(e -> true);
                Map<String, Integer> counts = new HashMap<>();
                int seen = 0;
                while (it != null && it.hasNext() && seen < maxScan) {
                    Entity e = it.next();
                    String dt = e.getDtClass() != null ? e.getDtClass().getDtName() : "<null>";
                    counts.put(dt, counts.getOrDefault(dt, 0) + 1);
                    seen++;
                }
                System.err.println("Scanned entities: " + seen);
                System.err.println("Distinct dt names (dt -> count):");
                counts.entrySet().stream()
                        .sorted((a,b)->Integer.compare(b.getValue(), a.getValue()))
                        .forEach(en -> System.err.println(en.getKey() + " -> " + en.getValue()));
                // Exit after printing
                System.exit(0);
            }
        });
    }
}