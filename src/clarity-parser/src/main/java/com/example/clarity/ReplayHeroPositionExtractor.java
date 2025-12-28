package com.example.clarity;

import skadistats.clarity.model.Entity;
import skadistats.clarity.processor.entities.Entities;
import skadistats.clarity.processor.entities.OnEntityCreated;
import skadistats.clarity.processor.entities.UsesEntities;
import skadistats.clarity.processor.gameevents.OnGameEvent;
import skadistats.clarity.processor.reader.OnTickStart;
import skadistats.clarity.processor.runner.Context;
import skadistats.clarity.processor.runner.SimpleRunner;
import skadistats.clarity.source.MappedFileSource;

import java.io.OutputStreamWriter;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

@UsesEntities
public class ReplayHeroPositionExtractor {

    private PrintWriter csvWriter;

    // Game time origin: prefer first lane creep spawn ("horn") so that game_time_sec is 0 at horn and negative before
    private Double hornTimeSec = null;       // first lane creep spawn time (0:00 UI)
    private Double gameTimeOriginSec = null; // equals hornTimeSec when detected
    private Double preGameStartSec = null;   // hornTimeSec - 60 (optional reference)
    private boolean gameTimeInitialized = false;

    private static final int SAMPLE_TICKS = 10;

    // Player metadata: slot -> {playerName, team, steamId, accountId, heroHandle}
    private final Map<Integer, Map<String, Object>> playerMetadata = new HashMap<>();
    private boolean metadataInitialized = false;

    // Optional event-based fallback calibration
    @OnGameEvent("state_changes")
    public void onGameStateChange(Context ctx, String name, Map<String, Object> keys) {
        // Keep as fallback if horn detection fails; we still prefer horn so clock is 0 at creep spawn
        Object newStateObj = keys.get("newState");
        if (newStateObj instanceof Number) {
            int newState = ((Number) newStateObj).intValue();
            if (!gameTimeInitialized && newState >= 3) {
                double t = ctx.getTick() / 30.0;
                // Do not set origin here if horn not seen yet; we just record pre-game as a hint
                preGameStartSec = t;
                System.out.printf("[GameTime] Event state=%d at tick=%d (pre-game hint at %.3f sec)%n",
                        newState, ctx.getTick(), t);
            }
        }
    }

    // Detect first lane creep spawn ("horn") and set game time origin to that moment.
    @OnEntityCreated
    public void onAnyEntityCreated(Context ctx, Entity e) {
        if (e == null || e.getDtClass() == null || gameTimeInitialized) return;
        String dtName = e.getDtClass().getDtName();
        // Lane creeps usually: CDOTA_BaseNPC_Creep_Lane_* / Siege creeps: CDOTA_BaseNPC_Creep_Siege
        boolean isLaneCreep = dtName.startsWith("CDOTA_BaseNPC_Creep_Lane")
                           || dtName.startsWith("CDOTA_BaseNPC_Creep_Siege")
                           || dtName.contains("Creep_Lane")
                           || dtName.contains("Creep_Siege");
        if (isLaneCreep) {
            hornTimeSec = ctx.getTick() / 30.0;
            gameTimeOriginSec = hornTimeSec;              // 0:00 at horn
            preGameStartSec = hornTimeSec - 60.0;         // optional reference: -60..0 before horn
            gameTimeInitialized = true;
            System.out.printf("[GameTime] Horn detected via creep spawn (%s) at tick=%d; horn=%.3f sec; pre-game start=%.3f sec%n",
                    dtName, ctx.getTick(), hornTimeSec, preGameStartSec);
        }
    }

    @OnTickStart
    public void onTick(Context ctx, boolean synthetic) {
        ensureOrUpdatePlayerMetadata(ctx);

        int tick = ctx.getTick();
        if (tick % SAMPLE_TICKS != 0) return;

        double replayTimeSec = tick / 30.0;
        Entities entities = ctx.getProcessor(Entities.class);
        if (entities == null) return;

        // Prefer direct gamerules game time if available; else use horn origin; else fallback to replay time
        double gameTimeSec = computeGameTimeSec(entities, replayTimeSec, tick);

        // Iterate all known player slots (from metadata) and resolve hero by handle
        for (Map.Entry<Integer, Map<String, Object>> entry : playerMetadata.entrySet()) {
            Integer slot = entry.getKey();
            Map<String, Object> meta = entry.getValue();

            Integer heroHandle = (Integer) meta.get("heroHandle");
            if (heroHandle == null) {
                Entity pr = entities.getByDtName("CDOTA_PlayerResource");
                if (pr != null) {
                    heroHandle = readHeroHandleForSlot(pr, slot);
                    if (heroHandle != null) {
                        meta.put("heroHandle", heroHandle);
                        System.out.printf("[Metadata] Updated HeroHandle for slot=%d -> %d%n", slot, heroHandle);
                    }
                }
            }
            if (heroHandle == null) continue;

            Entity hero = entities.getByHandle(heroHandle);
            if (hero == null || hero.getDtClass() == null) continue;

            String dtName = hero.getDtClass().getDtName();
            if (!dtName.startsWith("CDOTA_Unit_Hero_")) continue;

            // Skip illusions
            if (hero.hasProperty("m_bIsIllusion")) {
                Object isIllusion = hero.getProperty("m_bIsIllusion");
                if (isIllusion instanceof Boolean && (Boolean) isIllusion) continue;
            }

            // Position
            double x = getEntityPropertyAsDouble(hero, "CBodyComponent.m_vecX");
            double y = getEntityPropertyAsDouble(hero, "CBodyComponent.m_vecY");
            double z = getEntityPropertyAsDouble(hero, "CBodyComponent.m_vecZ");

            // Write CSV row
            writeCsvRow(
                replayTimeSec,
                gameTimeSec,
                tick,
                slot,
                String.valueOf(meta.getOrDefault("playerName", "Unknown")),
                String.valueOf(meta.getOrDefault("steamId", "Unavailable")),
                String.valueOf(meta.getOrDefault("accountId", "Unavailable")),
                getTeamName((Integer) meta.get("team")),
                dtName.replace("CDOTA_Unit_Hero_", ""),
                x, y, z
            );
        }
    }

    // Compute game_time_sec: prefer gamerules m_*GameTime, else horn origin, else replay_time_sec.
    private double computeGameTimeSec(Entities entities, double replayTimeSec, int tick) {
        Entity gr = entities.getByDtName("CDOTAGamerulesProxy");
        if (gr != null) {
            Number flGameTime = null;
            if (gr.hasProperty("m_flGameTime")) {
                flGameTime = (Number) gr.getProperty("m_flGameTime");
            } else if (gr.hasProperty("m_fGameTime")) {
                flGameTime = (Number) gr.getProperty("m_fGameTime");
            }
            if (flGameTime != null) {
                double gts = flGameTime.doubleValue();
                // Gamerules time is already the UI clock (negative before horn, 0 at horn)
                return gts;
            }
        }
        // If horn detected, use horn as origin so 0:00 at horn, negative before
        if (gameTimeOriginSec != null) {
            return replayTimeSec - gameTimeOriginSec;
        }
        // Fallback: if we only know pre-game start (rare), approximate using it
        if (preGameStartSec != null) {
            return replayTimeSec - preGameStartSec - 60.0; // makes 0 at horn if preGameStartSec was set at entry
        }
        // Last resort: same as replay time
        return replayTimeSec;
    }

    private void ensureOrUpdatePlayerMetadata(Context ctx) {
        Entities entities = ctx.getProcessor(Entities.class);
        if (entities == null) {
            System.out.println("[Metadata] Entities processor not available.");
            return;
        }

        Entity pr = entities.getByDtName("CDOTA_PlayerResource");
        if (pr == null) {
            System.out.println("[Metadata] PlayerResource not ready yet.");
            return;
        }

        if (!metadataInitialized) {
            System.out.println("[Metadata] Initializing PlayerResource data...");
        }

        for (int slot = 0; slot < 24; slot++) {
            Map<String, Object> data = playerMetadata.get(slot);
            if (data == null) {
                data = new HashMap<>();
                playerMetadata.put(slot, data);
            }

            // Player name: prefer vec path, fallback to legacy
            String playerName = null;
            if (pr.hasProperty("m_vecPlayerData." + slot + ".m_iszPlayerName")) {
                playerName = getEntityProperty(pr, "m_vecPlayerData." + slot + ".m_iszPlayerName");
            } else if (pr.hasProperty("m_iszPlayerNames." + slot)) {
                playerName = getEntityProperty(pr, "m_iszPlayerNames." + slot);
            }

            // Team: prefer vec path, fallback to legacy
            Integer team = null;
            if (pr.hasProperty("m_vecPlayerData." + slot + ".m_iPlayerTeam")) {
                team = getEntityProperty(pr, "m_vecPlayerData." + slot + ".m_iPlayerTeam");
            } else if (pr.hasProperty("m_iPlayerTeams." + slot)) {
                team = getEntityProperty(pr, "m_iPlayerTeams." + slot);
            }

            // Selected hero handle: prefer team data vec path; fallback to legacy array path
            Integer heroHandle = readHeroHandleForSlot(pr, slot);

            // Placeholder fields
            String steamId = (String) data.getOrDefault("steamId", "Unavailable");
            String accountId = (String) data.getOrDefault("accountId", "Unavailable");

            if (playerName != null) {
                boolean firstInit = !metadataInitialized && !data.containsKey("playerName");
                data.put("playerName", playerName);
                data.put("team", team);
                data.put("steamId", steamId);
                data.put("accountId", accountId);
                data.put("heroHandle", heroHandle);

                if (firstInit || heroHandle != null) {
                    System.out.printf("[Metadata] Slot=%d Name=%s Team=%s HeroHandle=%s%n",
                            slot, playerName, getTeamName(team), String.valueOf(heroHandle));
                }
            }
        }
        if (!metadataInitialized) {
            metadataInitialized = true;
            System.out.println("[Metadata] PlayerResource initialization complete.");
        }
    }

    private Integer readHeroHandleForSlot(Entity pr, int slot) {
        Integer heroHandle = null;
        if (pr.hasProperty("m_vecPlayerTeamData." + slot + ".m_hSelectedHero")) {
            heroHandle = getEntityProperty(pr, "m_vecPlayerTeamData." + slot + ".m_hSelectedHero");
        } else if (pr.hasProperty("m_hSelectedHero." + slot)) {
            heroHandle = getEntityProperty(pr, "m_hSelectedHero." + slot);
        }
        return heroHandle;
    }

    private void writeCsvRow(double replayTimeSec,
                             double gameTimeSec,
                             int tick,
                             int playerSlot,
                             String playerName,
                             String steamId,
                             String accountId,
                             String teamName,
                             String heroName,
                             double x,
                             double y,
                             double z) {
        String row = String.format(Locale.US,
                "%.3f,%.3f,%d,%d,%s,%s,%s,%s,%s,%.6f,%.6f,%.6f",
                replayTimeSec, gameTimeSec, tick, playerSlot,
                escapeCsv(playerName),
                escapeCsv(steamId),
                escapeCsv(accountId),
                escapeCsv(teamName),
                escapeCsv(heroName),
                x, y, z);
        csvWriter.print(row);
        csvWriter.print('\n');
        csvWriter.flush();
    }

    private String escapeCsv(String s) {
        if (s == null) return "";
        boolean mustQuote = s.contains(",") || s.contains("\"") || s.contains("\n") || s.contains("\r");
        if (!mustQuote) return s;
        String escaped = s.replace("\"", "\"\"");
        return "\"" + escaped + "\"";
    }

    private String getTeamName(Integer team) {
        if (team == null) return "Unknown";
        switch (team) {
            case 2: return "Radiant";
            case 3: return "Dire";
            default: return "Unknown";
        }
    }

    private double getEntityPropertyAsDouble(Entity entity, String property) {
        Object value = entity.getProperty(property);
        return value instanceof Number ? ((Number) value).doubleValue() : 0.0;
    }

    @SuppressWarnings("unchecked")
    private <T> T getEntityProperty(Entity entity, String property) {
        return entity.hasProperty(property) ? (T) entity.getProperty(property) : null;
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: java com.example.clarity.ReplayHeroPositionExtractor <replay.dem> <output.csv>");
            System.exit(1);
        }

        String replayFile = args[0];
        String outputCsv = args[1];

        ReplayHeroPositionExtractor extractor = new ReplayHeroPositionExtractor();
        extractor.csvWriter = new PrintWriter(
                new OutputStreamWriter(new FileOutputStream(outputCsv), StandardCharsets.UTF_8), true);
        extractor.csvWriter.println("replay_time_sec,game_time_sec,tick,player_slot,player_name,steam_id,account_id,team_name,hero,x,y,z");

        SimpleRunner runner = new SimpleRunner(new MappedFileSource(replayFile));
        runner.runWith(extractor);

        extractor.csvWriter.close();
        System.out.printf("Replay data written to %s%n", outputCsv);
    }
}