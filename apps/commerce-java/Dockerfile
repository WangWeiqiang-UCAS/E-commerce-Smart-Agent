FROM maven:3.9-eclipse-temurin-17 AS build
WORKDIR /workspace
COPY pom.xml ./
COPY mall-common/pom.xml mall-common/pom.xml
COPY mall-mbg/pom.xml mall-mbg/pom.xml
COPY mall-security/pom.xml mall-security/pom.xml
COPY mall-portal/pom.xml mall-portal/pom.xml
COPY mall-common mall-common
COPY mall-mbg mall-mbg
COPY mall-security mall-security
COPY mall-portal mall-portal
RUN mvn -pl mall-portal -am package -DskipTests

FROM eclipse-temurin:17-jre
RUN apt-get update \
    && apt-get install -y --no-install-recommends wget \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /workspace/mall-portal/target/mall-portal-1.0-SNAPSHOT.jar app.jar
EXPOSE 8085
ENTRYPOINT ["java", "-jar", "/app/app.jar"]
