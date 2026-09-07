package com.macro.mall.portal.service.impl;

import com.macro.mall.common.exception.Asserts;
import com.macro.mall.model.UmsMember;
import com.macro.mall.portal.config.ObjectStorageProperties;
import com.macro.mall.portal.domain.AfterSaleEvidence;
import com.macro.mall.portal.service.AfterSaleEvidenceService;
import com.macro.mall.portal.service.UmsMemberService;
import io.minio.BucketExistsArgs;
import io.minio.GetPresignedObjectUrlArgs;
import io.minio.MakeBucketArgs;
import io.minio.MinioClient;
import io.minio.PutObjectArgs;
import io.minio.http.Method;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.io.InputStream;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

/** Private evidence storage with owner-scoped object keys and no client URLs. */
@Service
public class AfterSaleEvidenceServiceImpl implements AfterSaleEvidenceService {
    private static final long MAX_FILE_SIZE = 5L * 1024 * 1024;
    private final JdbcTemplate jdbcTemplate;
    private final UmsMemberService memberService;
    private final ObjectProvider<MinioClient> minioClientProvider;
    private final ObjectStorageProperties storage;

    public AfterSaleEvidenceServiceImpl(JdbcTemplate jdbcTemplate, UmsMemberService memberService,
                                        ObjectProvider<MinioClient> minioClientProvider,
                                        ObjectStorageProperties storage) {
        this.jdbcTemplate = jdbcTemplate;
        this.memberService = memberService;
        this.minioClientProvider = minioClientProvider;
        this.storage = storage;
    }

    @Override
    public AfterSaleEvidence.UploadResponse upload(MultipartFile file) {
        UmsMember member = requiredMember();
        String contentType = validateFile(file);
        String extension = "image/jpeg".equals(contentType) ? "jpg" : "image/png".equals(contentType) ? "png" : "pdf";
        String objectKey = "after-sales/" + member.getId() + "/" + UUID.randomUUID() + "." + extension;
        MinioClient minio = requiredMinio();
        try {
            ensureBucket(minio);
            try (InputStream input = file.getInputStream()) {
                minio.putObject(PutObjectArgs.builder().bucket(storage.getBucket()).object(objectKey)
                        .stream(input, file.getSize(), -1).contentType(contentType).build());
            }
            jdbcTemplate.update("INSERT INTO oms_after_sale_attachment (member_id,object_key,content_type,content_length,created_at) "
                            + "VALUES (?,?,?,?,NOW())", member.getId(), objectKey, contentType, file.getSize());
            return new AfterSaleEvidence.UploadResponse(objectKey, contentType, file.getSize());
        } catch (Exception error) {
            throw new IllegalStateException("Unable to store after-sales evidence", error);
        }
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void attach(Long afterSaleId, AfterSaleEvidence.AttachRequest request) {
        UmsMember member = requiredMember();
        requireOwnedAfterSale(afterSaleId, member.getId());
        String expectedPrefix = "after-sales/" + member.getId() + "/";
        for (String objectKey : request.objectKeys()) {
            if (!objectKey.startsWith(expectedPrefix)) {
                Asserts.fail("Evidence object does not belong to the current member");
            }
            List<java.util.Map<String, Object>> attached = jdbcTemplate.queryForList("SELECT after_sale_id FROM oms_after_sale_attachment "
                    + "WHERE object_key=? AND member_id=?", objectKey, member.getId());
            if (attached.isEmpty()) {
                Asserts.fail("Evidence object was not uploaded by the current member");
            }
            Object linked = attached.get(0).get("after_sale_id");
            Long linkedAfterSaleId = linked instanceof Number ? ((Number) linked).longValue() : null;
            if (linkedAfterSaleId != null && !afterSaleId.equals(linkedAfterSaleId)) {
                Asserts.fail("Evidence object is already attached to another after-sales application");
            }
            jdbcTemplate.update("UPDATE oms_after_sale_attachment SET after_sale_id=?, attached_at=NOW() "
                    + "WHERE object_key=? AND member_id=? AND (after_sale_id IS NULL OR after_sale_id=?)",
                    afterSaleId, objectKey, member.getId(), afterSaleId);
        }
        List<String> objectKeys = jdbcTemplate.queryForList("SELECT object_key FROM oms_after_sale_attachment "
                + "WHERE after_sale_id=? AND member_id=? ORDER BY id", String.class, afterSaleId, member.getId());
        jdbcTemplate.update("UPDATE oms_order_return_apply SET proof_pics=? WHERE id=?", String.join(",", objectKeys), afterSaleId);
    }

    @Override
    public AfterSaleEvidence.DownloadResponse download(Long afterSaleId, String objectKey) {
        UmsMember member = requiredMember();
        requireOwnedAfterSale(afterSaleId, member.getId());
        Integer allowed = jdbcTemplate.queryForObject("SELECT COUNT(*) FROM oms_after_sale_attachment "
                + "WHERE after_sale_id=? AND member_id=? AND object_key=?", Integer.class,
                afterSaleId, member.getId(), objectKey);
        if (allowed == null || allowed != 1) {
            Asserts.fail("Evidence object not found");
        }
        try {
            String url = requiredMinio().getPresignedObjectUrl(GetPresignedObjectUrlArgs.builder()
                    .method(Method.GET).bucket(storage.getBucket()).object(objectKey).expiry(5, TimeUnit.MINUTES).build());
            return new AfterSaleEvidence.DownloadResponse(url);
        } catch (Exception error) {
            throw new IllegalStateException("Unable to create evidence download URL", error);
        }
    }

    private UmsMember requiredMember() {
        UmsMember member = memberService.getCurrentMember();
        if (member == null || member.getId() == null) {
            Asserts.fail("Member authentication is required");
        }
        return member;
    }

    private void requireOwnedAfterSale(Long afterSaleId, Long memberId) {
        Integer count = jdbcTemplate.queryForObject("SELECT COUNT(*) FROM oms_order_return_apply a "
                + "JOIN oms_order o ON o.id=a.order_id WHERE a.id=? AND o.member_id=?", Integer.class, afterSaleId, memberId);
        if (count == null || count != 1) {
            Asserts.fail("After-sales application not found");
        }
    }

    private MinioClient requiredMinio() {
        MinioClient minio = minioClientProvider.getIfAvailable();
        if (minio == null) {
            throw new IllegalStateException("Object storage is not configured");
        }
        return minio;
    }

    private void ensureBucket(MinioClient minio) throws Exception {
        if (!minio.bucketExists(BucketExistsArgs.builder().bucket(storage.getBucket()).build())) {
            minio.makeBucket(MakeBucketArgs.builder().bucket(storage.getBucket()).build());
        }
    }

    private String validateFile(MultipartFile file) {
        if (file == null || file.isEmpty() || file.getSize() > MAX_FILE_SIZE) {
            Asserts.fail("Evidence file must be between 1 byte and 5 MiB");
        }
        byte[] header;
        try (InputStream input = file.getInputStream()) {
            header = input.readNBytes(8);
        } catch (Exception error) {
            throw new IllegalStateException("Unable to inspect evidence file", error);
        }
        String detected = isJpeg(header) ? "image/jpeg" : isPng(header) ? "image/png" : isPdf(header) ? "application/pdf" : null;
        if (detected == null || !detected.equals(file.getContentType())) {
            Asserts.fail("Only JPEG, PNG, and PDF evidence files are accepted");
        }
        return detected;
    }

    private boolean isJpeg(byte[] header) {
        return header.length >= 3 && header[0] == (byte) 0xFF && header[1] == (byte) 0xD8 && header[2] == (byte) 0xFF;
    }

    private boolean isPng(byte[] header) {
        return header.length >= 8 && Arrays.equals(header, new byte[]{(byte) 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A});
    }

    private boolean isPdf(byte[] header) {
        return header.length >= 5 && header[0] == '%' && header[1] == 'P' && header[2] == 'D' && header[3] == 'F' && header[4] == '-';
    }
}
