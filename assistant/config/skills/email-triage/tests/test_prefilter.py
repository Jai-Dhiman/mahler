import sys
import unittest

sys.path.insert(0, "scripts")

from prefilter import is_noise


def _make_email(**kwargs):
    base = {
        "message_id": "test-id-001",
        "source": "gmail",
        "from_addr": "sender@example.com",
        "subject": "Hello",
        "received_at": "2026-04-07T10:00:00Z",
        "body_preview": "Hi there",
        "is_junk_rescue": False,
        "headers": {},
    }
    base.update(kwargs)
    return base


class TestIsNoiseReturnsTrue(unittest.TestCase):

    def test_list_unsubscribe_header(self):
        email = _make_email(headers={"list-unsubscribe": "<mailto:unsub@example.com>"})
        self.assertTrue(is_noise(email))

    def test_precedence_bulk(self):
        email = _make_email(headers={"precedence": "bulk"})
        self.assertTrue(is_noise(email))

    def test_precedence_list(self):
        email = _make_email(headers={"precedence": "list"})
        self.assertTrue(is_noise(email))

    def test_precedence_junk(self):
        email = _make_email(headers={"precedence": "junk"})
        self.assertTrue(is_noise(email))

    def test_precedence_case_insensitive(self):
        email = _make_email(headers={"precedence": "BULK"})
        self.assertTrue(is_noise(email))

    def test_auto_submitted_auto_generated(self):
        email = _make_email(headers={"auto-submitted": "auto-generated"})
        self.assertTrue(is_noise(email))

    def test_auto_submitted_auto_replied(self):
        email = _make_email(headers={"auto-submitted": "auto-replied"})
        self.assertTrue(is_noise(email))

    def test_from_noreply(self):
        email = _make_email(from_addr="noreply@service.com")
        self.assertTrue(is_noise(email))

    def test_from_no_reply_hyphen(self):
        email = _make_email(from_addr="no-reply@service.com")
        self.assertTrue(is_noise(email))

    def test_from_no_reply_underscore(self):
        email = _make_email(from_addr="no_reply@service.com")
        self.assertTrue(is_noise(email))

    def test_from_donotreply(self):
        email = _make_email(from_addr="donotreply@company.com")
        self.assertTrue(is_noise(email))

    def test_from_mailer_daemon(self):
        email = _make_email(from_addr="mailer-daemon@mail.example.com")
        self.assertTrue(is_noise(email))

    def test_from_postmaster(self):
        email = _make_email(from_addr="postmaster@example.com")
        self.assertTrue(is_noise(email))

    def test_from_noreply_display_name_format(self):
        email = _make_email(from_addr="Service Team <noreply@service.com>")
        self.assertTrue(is_noise(email))

    def test_from_linkedin(self):
        email = _make_email(from_addr="jobs@linkedin.com")
        self.assertTrue(is_noise(email))

    def test_from_indeed(self):
        email = _make_email(from_addr="alerts@indeed.com")
        self.assertTrue(is_noise(email))

    def test_from_glassdoor(self):
        email = _make_email(from_addr="noreply@glassdoor.com")
        self.assertTrue(is_noise(email))

    def test_from_ziprecruiter(self):
        email = _make_email(from_addr="jobs@ziprecruiter.com")
        self.assertTrue(is_noise(email))

    def test_from_monster(self):
        email = _make_email(from_addr="alerts@monster.com")
        self.assertTrue(is_noise(email))

    def test_from_dice(self):
        email = _make_email(from_addr="jobs@dice.com")
        self.assertTrue(is_noise(email))

    def test_from_careerbuilder(self):
        email = _make_email(from_addr="alerts@careerbuilder.com")
        self.assertTrue(is_noise(email))

    def test_from_lever_co(self):
        email = _make_email(from_addr="no-reply@lever.co")
        self.assertTrue(is_noise(email))

    def test_from_greenhouse_io(self):
        email = _make_email(from_addr="applications@greenhouse.io")
        self.assertTrue(is_noise(email))

    def test_from_workday(self):
        email = _make_email(from_addr="recruiting@workday.com")
        self.assertTrue(is_noise(email))

    def test_from_smartrecruiters(self):
        email = _make_email(from_addr="jobs@smartrecruiters.com")
        self.assertTrue(is_noise(email))

    def test_from_job_board_display_name_format(self):
        email = _make_email(from_addr="LinkedIn Jobs <jobs@linkedin.com>")
        self.assertTrue(is_noise(email))

    def test_subject_job_alert(self):
        email = _make_email(subject="Job Alert: Python Developer Roles")
        self.assertTrue(is_noise(email))

    def test_subject_new_jobs_for_you(self):
        email = _make_email(subject="New jobs for you in San Francisco")
        self.assertTrue(is_noise(email))

    def test_subject_jobs_matching(self):
        email = _make_email(subject="Jobs matching your profile")
        self.assertTrue(is_noise(email))

    def test_subject_recommended_jobs(self):
        email = _make_email(subject="Recommended jobs this week")
        self.assertTrue(is_noise(email))

    def test_subject_jobs_you_might_like(self):
        email = _make_email(subject="Jobs you might like near you")
        self.assertTrue(is_noise(email))

    def test_subject_job_alert_case_insensitive(self):
        email = _make_email(subject="JOB ALERT: Senior Engineer")
        self.assertTrue(is_noise(email))


class TestIsNoiseReturnsFalse(unittest.TestCase):

    def test_bohemian_club_secretary(self):
        email = _make_email(from_addr="secretary@bohemianclub.org", subject="Club dinner next week")
        self.assertFalse(is_noise(email))

    def test_direct_email_no_special_headers(self):
        email = _make_email(
            from_addr="john@example.com",
            subject="Lunch tomorrow?",
            headers={},
        )
        self.assertFalse(is_noise(email))

    def test_cc_no_special_headers(self):
        # Being on CC alone is not a noise signal
        email = _make_email(
            from_addr="alice@company.com",
            subject="Re: Project update",
            headers={},
        )
        self.assertFalse(is_noise(email))

    def test_github_notifications_not_job_board(self):
        email = _make_email(
            from_addr="notifications@github.com",
            subject="[myrepo] New pull request opened",
            headers={},
        )
        self.assertFalse(is_noise(email))

    def test_application_confirmation_not_job_alert(self):
        email = _make_email(
            from_addr="recruiting@acmecorp.com",
            subject="Your application to Acme Corp",
            headers={},
        )
        self.assertFalse(is_noise(email))


if __name__ == "__main__":
    unittest.main()
